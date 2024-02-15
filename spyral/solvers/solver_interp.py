from .guess import Guess
from ..core.cluster import Cluster
from ..core.track_generator import TrackInterpolator, InitialState
from ..core.constants import QBRHO_2_P
from ..interpolate import LinearInterpolator

from spyral_utils.nuclear import NucleusData

from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
import numpy as np
import math


def interpolate_trajectory(
    fit_params: Parameters, interpolator: TrackInterpolator, ejectile: NucleusData
) -> LinearInterpolator | None:
    """Use the interpolation scheme to generate a trajectory from the given fit parameters.

    Parameters
    ----------
    fit_params: Parameters
        the set of lmfit Parameters
    interpolator: TrackInterpolator
        the interpolation scheme
    ejectile: NucleusData
        data for the particle being tracked

    Returns
    -------
    LinearInterpolator | None
        Returns a LinearInterpolator interpolating the x,y coordinates on z upon success. Upon failure (typically an out of bounds for the interpolation scheme) returns None.
    """
    state = InitialState()
    state.vertex_x = fit_params["vertex_x"].value
    state.vertex_y = fit_params["vertex_y"].value
    state.vertex_z = fit_params["vertex_z"].value
    momentum = QBRHO_2_P * (fit_params["brho"].value * float(ejectile.Z))
    state.kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    state.polar = fit_params["polar"].value
    state.azimuthal = fit_params["azimuthal"].value

    return interpolator.get_interpolated_trajectory(
        state.vertex_x,
        state.vertex_y,
        state.vertex_z,
        state.polar,
        state.azimuthal,
        state.kinetic_energy,
    )


def objective_function(
    fit_params: Parameters,
    x: np.ndarray,
    interpolator: TrackInterpolator,
    ejectile: NucleusData,
) -> np.ndarray:
    """Function to be minimized. Returns errors for data compared to estimated track.

    Parameters
    ----------
    fit_params: Parameters
        the set of lmfit Parameters
    x: ndarray
        the data to be fit (x,y,z) coordinates in meters
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    ejectile: NucleusData
        the data for the particle being tracked

    Returns
    -------
    ndarray
        the error between the estimate and the data
    """
    trajectory = interpolate_trajectory(fit_params, interpolator, ejectile)
    errors = np.full(len(x), 1.0e6)
    if trajectory is None:
        return errors
    xy = trajectory.interpolate(x[:, 2])
    valid_trajectory = xy[~np.isnan(xy[:, 0])]
    limit = len(valid_trajectory)
    errors[:limit] = np.linalg.norm(x[:limit, :2] - valid_trajectory, axis=1)
    return errors[:]


def create_params(
    guess: Guess, ejectile: NucleusData, interpolator: TrackInterpolator
) -> Parameters:
    """Create the lmfit parameters with appropriate bounds

    Parameters
    ----------
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used

    Returns
    -------
    Parameters
        the lmfit parameters with bounds
    """
    interp_min_momentum = math.sqrt(
        interpolator.ke_min * (interpolator.ke_min + 2.0 * ejectile.mass)
    )
    interp_max_momentum = math.sqrt(
        interpolator.ke_max * (interpolator.ke_max + 2.0 * ejectile.mass)
    )
    interp_min_brho = (interp_min_momentum / QBRHO_2_P) / ejectile.Z
    interp_max_brho = (interp_max_momentum / QBRHO_2_P) / ejectile.Z
    interp_min_polar = interpolator.polar_min * np.pi / 180.0
    interp_max_polar = interpolator.polar_max * np.pi / 180.0

    uncertainty_position_xy = 0.01
    uncertainty_position_z = 0.1
    uncertainty_brho = 1.0

    min_brho = guess.brho - uncertainty_brho * 2.0
    if min_brho < interp_min_brho:
        min_brho = interp_min_brho
    max_brho = guess.brho + uncertainty_brho * 2.0
    if max_brho > interp_max_brho:
        max_brho = interp_max_brho

    min_polar = interp_min_polar
    max_polar = interp_max_polar
    if guess.polar > np.pi * 0.5:
        min_polar = np.pi * 0.5
    else:
        max_polar = np.pi * 0.5

    min_azimuthal = guess.azimuthal - np.pi * 0.25
    max_azimuthal = guess.azimuthal + np.pi * 0.25

    min_x = guess.vertex_x * 0.001 - uncertainty_position_xy * 2.0
    max_x = guess.vertex_x * 0.001 + uncertainty_position_xy * 2.0
    min_y = guess.vertex_y * 0.001 - uncertainty_position_xy * 2.0
    max_y = guess.vertex_y * 0.001 + uncertainty_position_xy * 2.0
    min_z = guess.vertex_z * 0.001 - uncertainty_position_z * 2.0
    max_z = guess.vertex_z * 0.001 + uncertainty_position_z * 2.0

    fit_params = Parameters()
    fit_params.add("brho", guess.brho, min=min_brho, max=max_brho)
    fit_params.add("polar", guess.polar, min=min_polar, max=max_polar)
    fit_params.add("azimuthal", guess.azimuthal, min=min_azimuthal, max=max_azimuthal)
    fit_params.add("vertex_x", guess.vertex_x * 0.001, min=min_x, max=max_x, vary=False)
    fit_params.add("vertex_y", guess.vertex_y * 0.001, min=min_y, max=max_y, vary=False)
    fit_params.add("vertex_z", guess.vertex_z * 0.001, min=min_z, max=max_z, vary=True)
    return fit_params


# For testing, not for use in production
def fit_model_interp(
    cluster: Cluster,
    guess: Guess,
    interpolator: TrackInterpolator,
    ejectile: NucleusData,
) -> Parameters | None:
    """Used for jupyter notebooks examining the good-ness of the model

    Parameters
    ----------
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used

    Returns
    -------
    Parameters | None
        Returns the best fit Parameters upon success, or None upon failure
    """
    # We only use 90% of the trajectory data (closest to vertex).
    # Since sorted in z, need to trim 10% from one end depending on direction
    traj_data: np.ndarray
    if guess.polar > np.pi * 0.5:
        traj_len = int(len(cluster.data) * 0.1)
        traj_data = cluster.data[traj_len:, :3] * 0.001
    else:
        traj_len = int(len(cluster.data) * 0.9)
        traj_data = cluster.data[:traj_len, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None

    fit_params = create_params(guess, ejectile, interpolator)
    is_too_short = True
    depth = 0
    while is_too_short:
        depth += 1
        try:
            trajectory = interpolate_trajectory(fit_params, interpolator, ejectile)
        except Exception:
            return

        if depth > 50:
            return

        if trajectory is None:
            fit_params["brho"].value += fit_params["brho"].value * 0.1
            continue

        traj_xy = trajectory.interpolate(traj_data[:, 2])
        if np.any(np.isnan(traj_xy[:, 0])):
            fit_params["brho"].value += fit_params["brho"].value * 0.1
        else:
            is_too_short = False

    result: MinimizerResult = minimize(
        objective_function, fit_params, args=(traj_data, interpolator, ejectile)
    )
    print(fit_report(result))

    return result.params


def solve_physics_interp(
    cluster_index: int,
    cluster: Cluster,
    guess: Guess,
    interpolator: TrackInterpolator,
    ejectile: NucleusData,
    results: dict[str, list],
):
    """High level function to be called from the application.

    Takes the Cluster and fits a trajectory to it using the initial Guess. It then writes the results to the dictionary.

    Parameters
    ----------
    cluster_index: int
        Index of the cluster in the hdf5 scheme. Used only for debugging
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    results: dict[str, list]
        storage for results from the fitting, which will later be written as a dataframe.
    """
    # We only use 90% of the trajectory data (closest to vertex).
    # Since sorted in z, need to trim 10% from one end depending on direction
    traj_data: np.ndarray
    depth = 0
    if guess.polar > np.pi * 0.5:
        traj_len = int(len(cluster.data) * 0.1)
        traj_data = cluster.data[traj_len:, :3] * 0.001
    else:
        traj_len = int(len(cluster.data) * 0.9)
        traj_data = cluster.data[:traj_len, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return

    fit_params = create_params(guess, ejectile, interpolator)
    # Sometimes brho is far enough away that the estimated trajectory stops before the end of the data. To avoid this, pre-adjust brho to better fit
    is_too_short = True
    while is_too_short:
        trajectory = None
        try:
            trajectory = interpolate_trajectory(fit_params, interpolator, ejectile)
        except Exception:
            # This is a case where the data/guess is so messed up that there is no valid trajectory with that length/energy
            return

        if trajectory is None:
            fit_params["brho"].value += fit_params["brho"].value * 0.1
            continue

        if depth > 50:
            return

        traj_xy = trajectory.interpolate(traj_data[:, 2])

        if np.any(np.isnan(traj_xy[:, 0])):
            fit_params["brho"].value += fit_params["brho"].value * 0.01
        else:
            is_too_short = False
        depth += 1

    best_fit: MinimizerResult = minimize(
        objective_function, fit_params, args=(traj_data, interpolator, ejectile)
    )

    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    # Best fit values and uncertainties
    results["vertex_x"].append(best_fit.params["vertex_x"].value)
    results["vertex_y"].append(best_fit.params["vertex_y"].value)
    results["vertex_z"].append(best_fit.params["vertex_z"].value)
    results["brho"].append(best_fit.params["brho"].value)
    results["polar"].append(best_fit.params["polar"].value)
    results["azimuthal"].append(best_fit.params["azimuthal"].value)
    results["redchisq"].append(best_fit.redchi)

    # Sometimes fit is so bad uncertainties cannot be estimated
    if hasattr(best_fit, "uvars"):
        results["sigma_vx"].append(best_fit.uvars["vertex_x"].s)
        results["sigma_vy"].append(best_fit.uvars["vertex_y"].s)
        results["sigma_vz"].append(best_fit.uvars["vertex_z"].s)
        results["sigma_brho"].append(best_fit.uvars["brho"].s)
        results["sigma_polar"].append(best_fit.uvars["polar"].s)
        results["sigma_azimuthal"].append(best_fit.uvars["azimuthal"].s)
    else:
        results["sigma_vx"].append(1.0e6)
        results["sigma_vy"].append(1.0e6)
        results["sigma_vz"].append(1.0e6)
        results["sigma_brho"].append(1.0e6)
        results["sigma_polar"].append(1.0e6)
        results["sigma_azimuthal"].append(1.0e6)
