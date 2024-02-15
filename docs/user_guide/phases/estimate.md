# Estimating the Physics

Now that we formed our particle trajectory clusters, it is time to think about extracting our physical observables. In general we want the following

- The location at which the reaction occured (reaction vertex)
- The kinematic properties of the particle observed (kinetic energy, polar angle, azimuthal angle)
- Some identifiers of particle species  (dE/dx)

The sort of natural physics way to think of this is to generate a model trajectory using these parameters and the equations of motion of a particle in an electromagnetic field coupled to an energy loss model for the gas, and then determine the best possible set of parameters using some optimization method (i.e. least-squares fitting). But before we jump into the deep end there we need some way to get good initial guesses for these values, otherwise it is really unlikely that our optimizer will ever find a good minimum for such a complex problem.

That is the goal of the estimation phase: try to find some way to estimate a value for these key trajectory observables so that we have a good starting point for a fit.

## How it Works

The estimation phase is probably the most hard-to-read part of the Spyral framework. It is very short; the entirety of the code is in `spyral/core/estimator.py` and the function `estimate_physics`, and only just eclipses 100 lines of code in total. But it does a *lot* of things, and makes a lot of, shall we say, educated guesses about the behavior of the trajectory. In all honesty, it is best to read the code for this section to understand how it works.

First, we do some bad trajectory rejection. Trajectories are rejected based on the number of points and their near-ness to the beam axis. If trajectories have too few points or too many points in the beam region, we reject them. These parameters are exposed in the [configuration](../config/estimate.md).

Once we're confident we've idenfied a valid trajectory, its time to start estimating. First we try to see if the trajectory is going forward (towards the micromegas) or backward (toward the window) by calculating the average &rho; (distance to beam axis) at either end of the trajectory. Due to energy loss, the trajectories in-spiral, so the end with the smaller &rho; is the direction of travel, and the larger is the closest to the reaction vertex.

We then fit a circle to the first arc of the trajectory. This is done by fitting the region from the start of the trajectory to the point of furthest distance in &rho;. That circle fit is used to estimate the radius of curvature. The circle fit is also used to extract the reaction vertex X and Y coordinates by extrapolating the fit circle to the closest approach to the beam axis. Finally, we also fit a small segment of the beginning of the trajectory to with a line in &rho; vs. z. Using the extrapolated vertex X and Y, we can then use the linear fit to estimate a vertex Z. The polar angle is extracted from the slope of the line in &rho;-z; the azimuthal from the orientation of the vertex with respect to the center of the circle fit.

The radius of curvature (also &rho;) is proportional to the momentum from the equation for cyclotron motion

$$
\begin{align}
    qvB = \frac{mv^2}{\rho} \\
    B\rho = \frac{p}{q}
\end{align}
$$

The factor B&rho; is commonly referred to a the magnetic rigidity, which we use as a proxy for kinetic energy. The actual equation we use is

$$
    B\rho = \frac{B_{det} r_{circle}}{\sin(\theta_{polar})}
$$

where we divide by the sin of the polar angle as only the field perpendicular to the motion contributes to the cyclotron motion.

Finally, we extract the stopping power (dE/dx) by adding the integrated charge along the trajectory and dividing by the total path length. In this way we have extracted all of the relevant physics parameters.

## Plotting and Particle ID

Once estimation is done, you can check the progress of the work and see how well Spyral is doing from a physics perspective. This involves plotting the B&rho; vs. dE/dx relationship to examine particle groups as well as kinematics in the relationship B&rho; vs. &theta;. To help with this Spyral ships with a notebook `particle_id.ipynb` which will allow you to plot and gate on particle groups. To launch the notebook use the following command from the Spyral repo:

```bash
jupyter-lab --notebook-dir=./notebooks
```

Be sure your Spyral virtualenv is active to run this.

Some important notes about gate drawing

Gates are drawn using matplotlib. Settings can be altered within the notebook to change file names and other properties. The format for a particle ID gate is the following in JSON:

```json
{
    "name": "my_pid",
    "Z": 1,
    "A": 1,
    "vertices": [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
}
```

Here the gate is specified to be for protons. A particle ID gate is *required* to move on to the final phase. You do not need to use the notebook to make a particle ID gate. You simply need to have a JSON file of the correct format.

The notebook will typically need tweaking from experiment to experiment. The histogram ranges often need extending/shrinking, and the number of bins will vary depending on total statistics. It also gives users a chance to see some of the [spyral-utils](https://github.com/gwm17/spyral-utils/) package, which can be used outside of Spyral to do further analysis.

## Final Thoughts

Estimating, as the name implies, is the least precise of all of the phases. It doesn't have as many parameters as the others because it's not clear at the time of writing which parts change significantly from experiment to experiment. Feedback on this section is welcome; it remains one of the areas where improvement is most desired.

Estimating is fast, and should not represent a signficant bottleneck to analysis.
