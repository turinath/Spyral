from .core.config import Config
from .core.workspace import Workspace
from .phase_1 import phase_1
from .phase_2 import phase_2
from .phase_3 import phase_3
from .phase_4_interp import phase_4_interp

from time import time
from multiprocessing import SimpleQueue
import datetime

def run_spyral(config: Config, queue: SimpleQueue):

    ws = Workspace(config.workspace)
    pad_map = ws.get_pad_map()
    nuclear_map = ws.get_nuclear_map()
    # start = time()

    for idx in range(config.run.run_min, config.run.run_max + 1, 1):

        # print(f'Analyzing run {idx}...')

        if config.run.do_phase1:
            phase_1(idx, ws, pad_map, config.trace, config.frib, config.cross, config.detector, queue)

        if config.run.do_phase2:
            phase_2(idx, ws, config.cluster, queue)

        if config.run.do_phase3:
            phase_3(idx, ws, config.estimate, config.detector, queue)

        if config.run.do_phase4:
            phase_4_interp(idx, ws, config.solver, config.detector, nuclear_map, queue)
        
    # stop = time()
    # timestr = str(datetime.timedelta(seconds = stop - start))
    # print(f'Total ellapsed runtime: {stop - start}s (i.e {timestr})')
