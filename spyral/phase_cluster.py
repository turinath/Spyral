from .core.config import ClusterParameters
from .core.point_cloud import PointCloud
from .core.clusterize import form_clusters, join_clusters, cleanup_clusters
from .core.workspace import Workspace
from .parallel.status_message import StatusMessage, Phase
from .core.spy_log import spyral_warn, spyral_error, spyral_info

import h5py as h5
from multiprocessing import SimpleQueue


def phase_cluster(
    run: int, ws: Workspace, cluster_params: ClusterParameters, queue: SimpleQueue
):
    """Core loop of the clustering phase

    Take the point clouds and break them into clusters which should represent particle trajectories.
    First the data is run through the HDBSCAN algorithm to generate initial clusters. Clusters are then joined
    based on their overlap and charge to make trajectory clusters. The trajectory clusters are then cleaned and
    smoothed.

    Parameters
    ----------
    run: int
        The run number to be processed
    ws: Workspace
        The project Workspace
    cluster_params: ClusterParameters
        Configuration parameters controlling the clustering algorithm
    queue: SimpleQueue
        Communication channel back to the parent process
    """

    # Check that point clouds exist
    point_path = ws.get_point_cloud_file_path(run)
    if not point_path.exists():
        spyral_warn(
            __name__,
            f"Point cloud data does not exist for run {run} at phase 2. Skipping.",
        )
        return

    cluster_path = ws.get_cluster_file_path(run)

    point_file = h5.File(point_path, "r")
    cluster_file = h5.File(cluster_path, "w")

    cloud_group: h5.Group = point_file["cloud"]
    if not isinstance(cloud_group, h5.Group):
        spyral_error(__name__, f"Point cloud group not present in run {run}!")
        return

    min_event: int = cloud_group.attrs["min_event"]
    max_event: int = cloud_group.attrs["max_event"]
    cluster_group: h5.Group = cluster_file.create_group("cluster")
    cluster_group.attrs["min_event"] = min_event
    cluster_group.attrs["max_event"] = max_event

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    count = 0

    # Process the data
    for idx in range(min_event, max_event + 1):
        if count > flush_val:
            count = 0
            queue.put(StatusMessage(run, Phase.CLUSTER, 1))
        count += 1

        cloud_data: h5.Dataset | None = None
        try:
            cloud_data = cloud_group[f"cloud_{idx}"]
        except Exception:
            continue

        if cloud_data is None:
            continue

        cloud = PointCloud()
        cloud.load_cloud_from_hdf5_data(cloud_data[:].copy(), idx)

        clusters = form_clusters(cloud, cluster_params)
        joined = join_clusters(clusters, cluster_params)
        cleaned = cleanup_clusters(joined, cluster_params)

        # Each event can contain many clusters
        cluster_event_group = cluster_group.create_group(f"event_{idx}")
        cluster_event_group.attrs["nclusters"] = len(cleaned)
        cluster_event_group.attrs["ic_amplitude"] = cloud_data.attrs["ic_amplitude"]
        cluster_event_group.attrs["ic_centroid"] = cloud_data.attrs["ic_centroid"]
        cluster_event_group.attrs["ic_integral"] = cloud_data.attrs["ic_integral"]
        cluster_event_group.attrs["ic_multiplicity"] = cloud_data.attrs[
            "ic_multiplicity"
        ]
        for cidx, cluster in enumerate(cleaned):
            local_group = cluster_event_group.create_group(f"cluster_{cidx}")
            local_group.attrs["label"] = cluster.label
            local_group.create_dataset("cloud", data=cluster.data)

    spyral_info(__name__, "Phase 2 complete.")
