from .core.config import Config
from .core.workspace import Workspace
from .core.spy_log import init_spyral_logger_child, spyral_info, spyral_error
from .phase_pointcloud import phase_pointcloud
from .phase_pointcloud_legacy import phase_pointcloud_legacy
from .phase_cluster import phase_cluster
from .phase_estimate import phase_estimate
from .phase_solve import phase_solve

from spyral_utils.nuclear import NuclearDataMap

from multiprocessing import SimpleQueue


def run_spyral(
    config: Config, run_list: list[int], queue: SimpleQueue, process_id: int
):
    """Spyral main loop

    This is the function to run a single processor of Spyral.
    Typically called by run_spyral_parallel and spawned to a child process.

    Parameters
    ----------
    config: Config
        The project configuration
    run_list: list[int]
        The set of runs for this process
    queue: SimpleQueue
        A communication channel back to the parent process for monitoring progress
    """

    ws = Workspace(config.workspace, config.run.is_legacy)
    pad_map = ws.get_pad_map()
    nuclear_map = NuclearDataMap()

    init_spyral_logger_child(ws, process_id)

    for idx in run_list:
        spyral_info(__name__, f"Processing run {idx}")

        # Lock the processing behind a try so that execptions are handled gracefully
        try:
            if config.run.do_pointcloud and not config.run.is_legacy:
                spyral_info(__name__, "Running phase point cloud")
                phase_pointcloud(
                    idx, ws, pad_map, config.get, config.frib, config.detector, queue
                )
            elif config.run.do_pointcloud:
                spyral_info(__name__, "Running phase point cloud with legacy extension")
                phase_pointcloud_legacy(
                    idx, ws, pad_map, config.get, config.frib, config.detector, queue
                )

            if config.run.do_cluster:
                spyral_info(__name__, "Running phase cluster")
                phase_cluster(idx, ws, config.cluster, queue)

            if config.run.do_estimate:
                spyral_info(__name__, "Running phase estimate")
                phase_estimate(idx, ws, config.estimate, config.detector, queue)

            if config.run.do_solve:
                spyral_info(__name__, "Running phase solve")
                phase_solve(idx, ws, config.solver, config.detector, nuclear_map, queue)
        except Exception as e:
            spyral_error(__name__, f"Exception while processing run {idx}: {e}")
