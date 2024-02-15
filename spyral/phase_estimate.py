from .core.cluster import Cluster
from .core.config import DetectorParameters, EstimateParameters
from .core.estimator import estimate_physics
from .core.workspace import Workspace
from .parallel.status_message import StatusMessage, Phase
from .core.spy_log import spyral_error, spyral_warn, spyral_info

from polars import DataFrame
import h5py as h5
from multiprocessing.queues import SimpleQueue


def phase_estimate(
    run: int,
    ws: Workspace,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
    queue: SimpleQueue,
):
    """The core loop of the estimate phase

    Estimate physics parameters (Brho, polar angle, vertex position, etc) for clusters from the cluster phase. This phase serves two purposes which are related. The
    first is to estimate initial values for the upcoming solving phase where ODE solutions are fit to the data (hence the name estimate). The second is to generate
    parameters (Brho, dEdx) to be used in a particle ID, which is also necessary for the solving phase.

    Parameters
    ----------
    run: int
        The run number to be processed
    ws: Workspace
        The project Workspace
    estimate_params: EstimateParameters
        Configuration parameters controlling the estimation algorithm
    detector_params: DetectorParameters
        Configuration parameters for physical detector properties

    """

    # Check that clusters exist
    cluster_path = ws.get_cluster_file_path(run)
    if not cluster_path.exists():
        spyral_warn(
            __name__, f"Cluster file for run {run} not present for phase 3. Skipping."
        )
        return

    estimate_path = ws.get_estimate_file_path_parquet(run)

    cluster_file = h5.File(cluster_path, "r")
    cluster_group: h5.Group = cluster_file["cluster"]
    if not isinstance(cluster_group, h5.Group):
        spyral_error(__name__, f"Cluster group not present for run {run}!")
        return

    min_event: int = cluster_group.attrs["min_event"]
    max_event: int = cluster_group.attrs["max_event"]

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    count = 0

    # estimation results
    data: dict[str, list] = {
        "event": [],
        "cluster_index": [],
        "cluster_label": [],
        "ic_amplitude": [],
        "ic_centroid": [],
        "ic_integral": [],
        "ic_multiplicity": [],
        "vertex_x": [],
        "vertex_y": [],
        "vertex_z": [],
        "center_x": [],
        "center_y": [],
        "center_z": [],
        "polar": [],
        "azimuthal": [],
        "brho": [],
        "dEdx": [],
        "dE": [],
        "arclength": [],
        "direction": [],
        "eloss": [],
        "cutoff_index": [],
    }

    # Process data
    for idx in range(min_event, max_event + 1):
        if count > flush_val:
            count = 0
            queue.put(StatusMessage(run, Phase.ESTIMATE, 1))
        count += 1

        event: h5.Group | None = None
        try:
            event = cluster_group[f"event_{idx}"]
        except Exception:
            continue

        nclusters = event.attrs["nclusters"]
        ic_amp: float = event.attrs["ic_amplitude"]
        ic_cent: float = event.attrs["ic_centroid"]
        ic_int: float = event.attrs["ic_integral"]
        ic_mult: float = event.attrs["ic_multiplicity"]
        # Go through every cluster in each event
        for cidx in range(0, nclusters):
            local_cluster: h5.Group | None = None
            try:
                local_cluster = event[f"cluster_{cidx}"]
            except Exception:
                continue

            cluster = Cluster(
                idx, local_cluster.attrs["label"], local_cluster["cloud"][:].copy()
            )

            # Cluster is loaded do some analysis
            estimate_physics(
                cidx,
                cluster,
                ic_amp,
                ic_cent,
                ic_int,
                ic_mult,
                estimate_params,
                detector_params,
                data,
            )

    # Write the results to a DataFrame
    df = DataFrame(data)
    df.write_parquet(estimate_path)
    spyral_info(__name__, "Phase 3 complete.")
