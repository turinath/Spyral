# Solving for Physics

The final phase of Spyral is solving for the physical observables using our estimates from the estimation phase. Solving is a two step process outlined below:

1. Generate an interpolation mesh of ODE solutions
2. Fit the solutions to the data trajectory

Below we'll break down each of these steps

## Generating the Interpolation Mesh

Before Spyral does anything, it will check to see if you're planning to run the solve phase. If you are, it will check to see if the interpolation mesh you requested (see the [configuration](../config/solver.md)); if the mesh does not exist, it will generate one.

In the [configuration](../config/solver.md) you can specify the coarseness of the mesh; the particle species is taken from the given particle ID gate. Spyral walks through the kinetic energy, polar angle range at the steps needed for the number of bins requested, making a complete trajectory for each energy, angle. Care is taken to handle the time range of the solving. First, Spyral will walk through the phase space of the configuration and find the time range of the longest trajectory (in time) using an inital coarse time window of 1 &mu;s. Then Spyral generates the acutal mesh using this refined time window. The number of time steps is a configuration parameter and can be used to optimize resolution. Each step saves 3 double precision floats (8 bytes) for 24 bytes per point. Multiply this by the total number of bins (energy bins x angle bins x timesteps) to get the total size of the mesh in bytes. Even trajectories which stop before the 1.0 &mu;s time window have the same total data size. This will give you an idea of what the size of your mesh will be.

The mesh size is important for several reasons. First is, obviously, a finer mesh (more bins) gives better results, as it reduces the degree of interpolation. But since a finer mesh is bigger, the computational strain is larger. The interpolation is a bilinear interpolation; it requires the *entire* mesh be loaded into memory. This limits the size of the mesh to the amount of memory in your machine divided by the number of parallel processes.

Finally, when a trajectory is requested, Spyral interpolates on energy and angle to get a trajectory points or fewer (points where the particle stopped are trimmed).

When a trajectory is requested, the full energy, polar angle, azimuthal angle, and reaction vertex must be given. The azimuthal angle and vertex are used to rotate and translate the trajectory to the proper orientation (this reduces the phase space of the interpolation mesh).

In general, it has been found that 0.5 degree steps in polar angle are best, while 500 keV steps in energy are acceptable. However, this may need tuning from experiment to experiment. The interpolation code can be found in `spyral/core/track_generator.py` as well as `spyral/interpolate/`

## Fitting to Data

Spyral performs a least squares fit to data using the [lmfit](https://lmfit.github.io/lmfit-py/) library. Spyral uses the L-BFGS-B quasi-Newton minimization scheme to minimize the average error between the data and the trajectory. The average error is computed by looping over the data and the trajectory and calculating the smallest distance between the given data point and a point on the trajectory. These minimum distances are then averaged.

The best fit parameters and their uncertainties are written to disk in an Apache parquet file. The &chi;<sup>2</sup> and reduced &chi;<sup>2</sup> are also written.

Solver code can be found in `spyral/solvers/solver_interp.py`.

## Output

The output of the solver phase is a dataframe written to a parquet file in the `physics` directory of the workspace. Files are named by run and the particle species from the particle ID gate. The available values in the data frame are as follows:

- event: The event number associated with this data
- cluster_index: the cluster index associated with this data
- cluster_label: the cluster label associated with this data
- vertex_x: the fitted vertex x-coordinate
- sigma_vx: the error on the fitted vertex x-coordinate (unused) in meters
- vertex_y: the fitted vertex y-coordinate in meters
- sigma_vy: the error on the fitted vertex y-coordinate (unused) in meters
- vertex_z: the fitted vertex z-coordinate in meters
- sigma_vz: the error on the fitted vertex z-coordinate (unused) in meters
- brho: the fitted (total) B&rho; in Tm
- sigma_brho: the error on the fitted total B&rho; in Tm (unused)
- polar: the fitted polar angle of the trajectory in radians
- sigma_polar: the error on the fitted polar angle of the trajectory in radians (unused)
- azimuthal: the resulting azimuthal angle (not fitted) in radians. This calculated from the best fit vertex position.
- sigma_azimuthal: error on the resulting azimuthal angle (not fitted) in radians (unused)
- redchisq: the reduced &chi;<sup>2</sup> of the fit (error)

The error values are unused because the L-BFGS-B optimization scheme does not generally report parameter uncertainties.

## Final Thoughts

Generating a good interpolation mesh is key. If the mesh is too coarse the results will become equally coarse.

Through testing, it was found that the reaction vertex is *highly* correlated to the azimuthal and polar angles. As such, only the vertex z position is acutally fit; x and y are fixed. This greatly reduces the correlation. More testing should be done to look for alternatives.

To make performance reasonable [Numba](../numba.md) is used to just-in-time compile the interpolation scheme and objective function. Checkout that section of the docs for more details.

Other solvers have been used/tried. Of particular interest is using the Unscented Kalman Filter, which feels like a really great solution to our problem, but has some critical issues (uneven step size, gaps in trajectories, etc.). See `spyral/solvers/solver_kalman.py` if you're curious. Note that that code is out-of-date with the rest of Spyral. In the future it will most likely be relegated to an isolated branch of the repository.

An additional method of interest is least-squares fitting using a secondary interpolation on z from the trajectory. However, in practice, this was tested to give generally worse results than this method. But more concrete testing is needed to verify this behavior.

Solving is typically fast relative to  the longer phases like the point cloud phase, however, if the previous phases are not optimized well, the solver can take significantly longer due to noisy data.
