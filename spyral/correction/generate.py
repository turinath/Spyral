from ..core.config import DetectorParameters
from ..interpolate import BilinearInterpolator

import numpy as np
from pathlib import Path
import h5py as h5
from contourpy import contour_generator, ContourGenerator

def generate_electron_correction(garf_file_path: Path, output_path: Path, params: DetectorParameters):
    '''
    Need to convert garfield correction data into regularly spaced interpolation scheme
    over rho and z with the correct units

    Garfield data format: 
    [x_initial, y_initial, x_final, y_final, z_final, time] all in cm
    x=z, y=rho, x=transverse
    '''

    garfield_data: np.ndarray = np.loadtxt(garf_file_path, dtype=float)

    chunk_size = 55 # garfield data in 55 row chunks (steps in rho)
    chunk_midpoint_index = 27 # index of the chunk midpoint
    n_chunks = int(len(garfield_data) / chunk_size) # number of chunks (steps in z)

    z_steps = np.linspace(30.0, 1000.0, 98)
    gz_min = 30.0
    gz_max = 1000.0
    gz_bins = 98
    rho_steps = np.linspace(-270.0, 270.0, 55)
    grho_min = -270.0
    grho_max = 270.0
    grho_bins = 55

    rho_garf_points, z_garf_points = np.meshgrid(rho_steps, z_steps)

    rho_final: np.ndarray = np.zeros((n_chunks, chunk_size))
    misc_final: np.ndarray = np.zeros((n_chunks, chunk_size, 3))
    t_final: np.ndarray = np.zeros((n_chunks, chunk_size))
    for chunk in range(n_chunks):
        for row in range(chunk_size):
            #Convert distances to mm
            misc_final[chunk, row, 0] = garfield_data[chunk * chunk_size + row, 2] * 10.0 #z
            rho_final[chunk, row] = garfield_data[chunk * chunk_size + row, 3] * 10.0 #radial
            misc_final[chunk, row, 1] = garfield_data[chunk * chunk_size + row, 4] * 10.0 #transverse
            #Time in nanoseconds
            misc_final[chunk, row, 2] = garfield_data[chunk * chunk_size + row, 5] #time/z initial

    for chunk in misc_final:
        mid_val = chunk[chunk_midpoint_index, 2]
        chunk[:, 2] -= mid_val

    t_final *= params.detector_length / ((params.window_time_bucket - params.micromegas_time_bucket) * (1.0/params.get_frequency * 1000.0))

    interp = BilinearInterpolator(gz_min, gz_max, gz_bins, grho_min, grho_max, grho_bins, misc_final)
    contour = contour_generator(z_garf_points, rho_garf_points, rho_final)

    rho_bin_min = 0.0
    rho_bin_max = 275.0
    rho_bins = 276

    z_bin_min = 0.0
    z_bin_max = 1000.0
    z_bins = 1001

    rho_points = np.linspace(rho_bin_min, rho_bin_max, rho_bins)
    z_points = np.linspace(z_bin_min, z_bin_max, z_bins)

    correction_grid = np.zeros((276, 1001, 3))

    for ridx, r in enumerate(rho_points):
        for zidx, z in enumerate(z_points):
            #rescale z to garfield
            zg = (1.0 - z*0.001) * 970.0 + 30.0 #Garfield first point is at 30.0 mm
            rho_cor = interpolate_initial_rho(contour, z, r) - r
            correction_grid[ridx, zidx, 0] = rho_cor
            others = interp.interpolate(zg, rho_cor)
            correction_grid[ridx, zidx, 1] = others[1]
            correction_grid[ridx, zidx, 2] = others[2]

    outfile = h5.File(output_path, 'w')
    outfile.create_dataset('correction_grid', data=correction_grid)

def interpolate_initial_rho(contour: ContourGenerator, z: float, rho: float) -> float:
    lines = contour.lines(rho)
    if len(lines) == 0:
        return 0.0
    line: np.ndarray = lines[0]
    line = line[line[:, 0].argsort()]

    return np.interp(np.array([z]), line[:, 0], line[:, 1])[0]