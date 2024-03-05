from .core.config import Config
from .core.workspace import Workspace
from .core.track_generator import (
    generate_track_mesh,
    MeshParameters,
    check_mesh_needs_generation,
)
from .core.particle_id import load_particle_id
from .correction import generate_electron_correction
from .parallel.status_message import StatusMessage, Phase
from .parallel.run_stack import create_run_stacks
from .run import run_spyral
from .core.spy_log import init_spyral_logger_parent, spyral_info

from spyral_utils.nuclear import NuclearDataMap
from spyral_utils.nuclear.target import GasTarget, load_target

from multiprocessing import Process, SimpleQueue
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from time import time


def generate_shared_resources(config: Config):
    """Build the resources used by Spyral

    Shared resources such as interpolation tables need to be generated *before* the
    processors are started. Otherwise, there may be race conditions.

    Parameters
    ----------
    config: Config
        The project configuration
    """

    print("Checking to see if any shared resources need to be created...")
    ws = Workspace(config.workspace)
    # Get rid of any old log files
    ws.clear_log_path()
    # initialize our logger for the parent process
    init_spyral_logger_parent(ws)

    nuc_map = NuclearDataMap()

    if config.run.do_pointcloud and config.detector.do_garfield_correction:
        ecorr_path = ws.get_correction_file_path(
            Path(config.detector.garfield_file_path)
        )
        if not ecorr_path.exists():
            print(
                "Creating the electric field correction data... This may take some time..."
            )
            generate_electron_correction(
                Path(config.detector.garfield_file_path), ecorr_path, config.detector
            )
            print("Done.")

    if config.run.do_solve:
        target = load_target(Path(config.solver.gas_data_path), nuc_map)
        pid = load_particle_id(
            ws.get_gate_file_path(config.solver.particle_id_filename), nuc_map
        )
        if pid is None:
            print(
                "Could not create trajectory mesh, particle ID does not have the correct format!"
            )
            print("Particle ID is required for running the solver stage (phase 4).")
            raise Exception
        if not isinstance(target, GasTarget):
            print(
                "Could not create trajectory mesh, target data does not have the correct format for a GasTarget!"
            )
            print("Gas Target is required for running the solver stage (phase 4).")
            raise Exception
        mesh_params = MeshParameters(
            target,
            pid.nucleus,
            config.detector.magnetic_field,
            config.detector.electric_field,
            config.solver.n_time_steps,
            config.solver.interp_ke_min,
            config.solver.interp_ke_max,
            config.solver.interp_ke_bins,
            config.solver.interp_polar_min,
            config.solver.interp_polar_max,
            config.solver.interp_polar_bins,
        )
        track_path = ws.get_track_file_path(pid.nucleus, target)
        do_gen = check_mesh_needs_generation(track_path, mesh_params)
        if do_gen:
            print("Creating the trajectory mesh... This may take some time...")
            generate_track_mesh(mesh_params, track_path)
            print("Done.")
    print("Shared resources are ready.")


def run_spyral_parallel(config: Config, no_progress: bool = False):
    """Spyral's parent process function

    This is the main function to be called to run Spyral. The configuration will be
    used to generate a set of child processes to analyze the data range.

    Parameters
    ----------
    config: Config
        The project configuration
    no_progress: bool
        If this is True progress bars will not be displayed (default is False)
    """
    # For housekeeping, track and log how long the execution takes
    start = time()

    # Some data must be made and shared across processes through files.
    generate_shared_resources(config)
    n_processes = config.run.n_processes

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    stats: list[Phase] = []
    runs: list[int] = []

    print("Optimally generating run lists for each process...")
    stacks = create_run_stacks(config, n_processes)
    spyral_info(__name__, f"Run stacks: {stacks}")
    print("Done.")

    # Create the child processes
    for s in range(0, len(stacks)):
        local_config = deepcopy(config)
        queues.append(SimpleQueue())
        processes.append(
            Process(
                target=run_spyral,
                args=(local_config, stacks[s], queues[-1], s),
                daemon=False,
            )
        )
        pbars.append(tqdm(total=100, disable=no_progress))
        stats.append(Phase.WAIT)
        runs.append(-1)
        pbars[-1].set_description(f"| Process {s} | { str(stats[-1]) } |")

    for process in processes:
        process.start()

    anyone_alive: bool
    # main loop
    while True:
        anyone_alive = False
        # check processes still going
        for process in processes:
            if process.is_alive():
                anyone_alive = True
                break

        if not anyone_alive:
            break

        # Read events out of the queues
        for idx, q in enumerate(queues):
            if q.empty():
                continue

            msg: StatusMessage = q.get()
            if msg.phase != stats[idx] or msg.run != runs[idx]:
                pbars[idx].reset()
                pbars[idx].set_description(f"| Process {idx} | {msg}")
                stats[idx] = msg.phase
                runs[idx] = msg.run
            pbars[idx].update(msg.progress)

    # Shutdown

    for bar in pbars:
        bar.close()

    for q in queues:
        q.close()

    for process in processes:
        process.join()

    stop = time()
    duration = stop - start
    hours, sec = divmod(duration, 3600)
    minutes, sec = divmod(sec, 60)
    spyral_info(
        __name__, f"Total ellapsed time: {int(hours)} hrs {int(minutes)} min {sec:.4} s"
    )
