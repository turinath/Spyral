# Solving for Physics

The final phase of Spyral is solving for the physical observables using our estimates from the estimation phase. Solving is a two step process outlined below:

1. Generate an interpolation mesh of ODE solutions
2. Fit the solutions to the data trajectory

Below we'll break down each of these steps

## Generating the Interpolation Mesh

Before Spyral does anything, it will check to see if you're planning to run the solve phase. If you are, it will check to see if the interpolation mesh you requested (see the [configuration](../config/solver.md)); if the mesh does not exist, it will generate one.

In the [configuration](../config/solver.md) you can specify the coarseness of the mesh; the particle species is taken from the given particle ID gate. Spyral walks through the kinetic energy, polar angle range at the steps needed for the number of bins requested, making a complete trajectory for each energy, angle. The time window range for the ODE solver is 1 &mu;s. The number of time steps is a configuration parameter and can be used to optimize resolution. Each step saves 3 double precision floats (8 bytes) for 24 bytes per point. Multiply this by the total number of bins (energy bins x angle bins x timesteps) to get the total size of the mesh in bytes. Even trajectories which stop before the 1.0 &mu;s time window have the same total data size. This will give you an idea of what the size of your mesh will be.

The mesh size is important for several reasons. First is, obviously, a finer mesh (more bins) gives better results, as it reduces the degree of interpolation. But since a finer mesh is bigger, the computational strain is larger. The interpolation is a bilinear interpolation; it requires the *entire* mesh be loaded into memory. This limits the size of the mesh to the amount of memory in your machine divided by the number of parallel processes.

Finally, when a trajectory is requested, Spyral interpolates on energy and angle to get a trajectory of 500 points or fewer (points where the particle stopped are trimmed). Spyral then interpolates the trajectory in z, returning an interpolated (x,y) for a given z value.

When a trajectory is requested, the full energy, polar angle, azimuthal angle, and reaction vertex must be given. The azimuthal angle and vertex are used to rotate and translate the trajectory to the proper orientation (this reduces the phase space of the interpolation mesh).

In general, it has been found that 0.5 degree steps in polar angle are best, while 500 keV steps in energy are acceptable. However, this may need tuning from experiment to experiment. The interpolation code can be found in `spyral/core/track_generator.py` as well as `spyral/interpolate/`

## Fitting to Data

Spyral performs a least squares fit to data using the [lmfit](https://lmfit.github.io/lmfit-py/) library. Based on the estimated parameters, Spyral requests a trajectory from the interpolator. For each data point, the interpolator calculates (x,y) from the data z. The difference between data (x,y) and interpolator (x,y) is minimized to find the best parameters.

Special care is taken to handle cases where a bad B&rho; guess leads to an interpolated trajectory which is too short to accomidate the data. At the start of solving the interpolated trajectory is too checked to see if it is too short; if it is B&rho; is slowly "wiggled" until the trajectory is long enough. Once fitting begins, if the minimizer walks to energies which result in too small trajectories, the points for which there is no interpolation value are given a large error (1.0e6). This biases the minimizer to pick trajectories which are long enough to cover all of the data.

The best fit parameters and their uncertainties are written to disk in an Apache parquet file. The &chi;<sup>2</sup> and reduced &chi;<sup>2</sup> are also written.

Solver code can be found in `spyral/solvers/solver_interp.py`.

## Final Thoughts

Generating a good interpolation mesh is key. If the mesh is too coarse the results will become equally coarse.

Through testing, it was found that the reaction vertex is *highly* correlated to the azimuthal and polar angles. As such, only the vertex z position is acutally fit; x and y are fixed. This greatly reduces the correlation. More testing should be done to look for alternatives.

To make performance reasonable [Numba](../numba.md) is used to just-in-time compile the interpolation scheme. Checkout that section of the docs for more details.

Other solvers have been used/tried. Of particular interest is using the Unscented Kalman Filter, which feels like a really great solution to our problem, but has some critical issues (uneven step size, gaps in trajectories, etc.). See `spyral/solvers/solver_kalman.py` if you're curious. Note that that code is out-of-date with the rest of Spyral. In the future it will most likely be relegated to an isolated branch of the repository.

Solving is very fast, typically the fastest phase in Spyral. Thanks Numba!
