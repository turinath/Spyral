from .cluster import Cluster
from .config import DetectorParameters, EstimateParameters
from ..geometry.circle import generate_circle_points, least_squares_circle
from .spy_log import spyral_warn

import numpy as np
import math
from scipy.signal import argrelmax
from scipy.stats import linregress
from scipy.interpolate import CubicSpline
from enum import Enum


class Direction(Enum):
    """Enum for the direction of a trajectory

    Attributes
    ----------
    NONE: int
        Invalid value (-1)
    FORWARD: int
        Trajectory traveling in the positive z-direction (0)
    BACKWARD: int
        Trajectory traveling in the negative z-direction (1)
    """

    NONE: int = -1
    FORWARD: int = 0
    BACKWARD: int = 1


def estimate_physics(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
    results: dict[str, list],
):
    # Run estimation where we attempt to guess the right direction
    is_good, direction = estimate_physics_pass(
        cluster_index,
        cluster,
        ic_amplitude,
        ic_centroid,
        ic_integral,
        ic_multiplicity,
        estimate_params,
        detector_params,
        results,
    )

    # If estimation was consistent or didn't meet valid criteria were done
    if is_good or (not is_good and direction == Direction.NONE):
        return
    # If we made a bad guess, try the other direction
    elif direction == Direction.FORWARD:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            estimate_params,
            detector_params,
            results,
            Direction.BACKWARD,
        )
    else:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            estimate_params,
            detector_params,
            results,
            Direction.FORWARD,
        )


def choose_direction(cluster: Cluster) -> Direction:
    rhos = np.linalg.norm(cluster.data[:, :2], axis=1)  # cylindrical coordinates rho
    direction = Direction.NONE

    # See if in-spiraling to the window or microgmegas, sets the direction and guess of z-vertex
    if rhos[0] < rhos[-1]:
        direction = Direction.FORWARD
    else:
        direction = Direction.BACKWARD

    return direction


def estimate_physics_pass(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
    results: dict[str, list],
    chosen_direction: Direction = Direction.NONE,
) -> tuple[bool, Direction]:
    """Estimate the physics parameters for a cluster which could represent a particle trajectory

    Estimation is an imprecise process (by definition), and as such this algorithm requires a lot of
    manipulation of the data.

    Parameters
    ----------
    cluster_index: int
        The cluster index in the HDF5 file.
    cluster: Cluster
        The cluster to estimate
    ic_amplitude:
        The ion chamber amplitude for this cluster
    ic_centroid:
        The ion chamber centroid for this cluster
    ic_integral:
        The ion chamber integral for this cluster
    estimate_params:
        Configuration parameters controlling the estimation algorithm
    detector_params:
        Configuration parameters for the physical detector properties
    results: dict[str, int]
        Dictionary to store estimation results in

    """
    # Do some cleanup, reject clusters which have too few points
    if len(cluster.data) < estimate_params.min_total_trajectory_points:
        return (False, Direction.NONE)
    beam_region_fraction = float(
        len(
            cluster.data[
                np.linalg.norm(cluster.data[:, :2], axis=1)
                < detector_params.beam_region_radius
            ]
        )
    ) / float(len(cluster.data))
    if beam_region_fraction > 0.9:
        return (False, Direction.NONE)

    direction = chosen_direction
    vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    center = np.array([0.0, 0.0, 0.0])  # spiral center
    # If chosen direction is set to NONE, we want to have the algorithm
    # try to decide which direction the trajectory is going
    if direction == Direction.NONE:
        direction = choose_direction(cluster)

    if direction == Direction.BACKWARD:
        cluster.data = np.flip(cluster.data, axis=0)

    # Guess that the vertex is the first point; make sure to copy! not reference
    vertex[:] = cluster.data[0, :3]

    # Find the first point that is furthest from the vertex in rho (maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster.data[1:, :2] - vertex[:2], axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster.data[: (maximum + 1)]

    # Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(
        first_arc[:, 0], first_arc[:, 1]
    )
    # better_radius = np.linalg.norm(cluster.data[0, :2] - center[:2])
    circle = generate_circle_points(center[0], center[1], radius)
    # Re-estimate vertex using the fitted circle. Extrapolate back to point closest to beam axis
    vertex_estimate_index = np.argsort(np.linalg.norm(circle, axis=1))[0]
    vertex[:2] = circle[vertex_estimate_index]
    # Re-calculate distance to vertex, maximum, first arc
    rho_to_vertex = np.linalg.norm((cluster.data[:, :2] - vertex[:2]), axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster.data[: (maximum + 1)]

    # Do a linear fit to small segment of trajectory to extract rho vs. z and extrapolate vertex z
    test_index = max(10, int(maximum * 0.5))
    # test_index = 10
    fit = linregress(cluster.data[:test_index, 2], rho_to_vertex[:test_index])
    vertex_rho = np.linalg.norm(vertex[:2])
    vertex[2] = (vertex_rho - fit.intercept) / fit.slope
    center[2] = vertex[2]

    # Toss tracks whose verticies are not close to the origin in x,y
    if vertex_rho > estimate_params.max_distance_from_beam_axis:
        return (False, Direction.NONE)

    polar = math.atan(fit.slope)
    # We have a self consistency case here. Polar should match chosen Direction
    if (polar > 0.0 and direction == Direction.BACKWARD) or (
        polar < 0.0 and direction == Direction.FORWARD
    ):
        return (
            False,
            direction,
        )  # Our direction guess was bad, we need to try again with the other direction
    elif direction is Direction.BACKWARD:
        polar += math.pi

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal -= math.pi * 1.5
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi

    brho = (
        detector_params.magnetic_field * radius * 0.001 / np.abs(math.sin(polar))
    )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0

    arclength = 0.0
    charge_deposited = first_arc[0, 3]
    for idx in range(len(first_arc) - 1):
        # Stop integrating if we leave the small pad region
        if np.linalg.norm(first_arc[idx + 1, :2]) > 152.0:
            break
        arclength += np.linalg.norm(first_arc[idx + 1, :3] - first_arc[idx, :3])
        charge_deposited += first_arc[idx + 1, 3]
    if arclength == 0.0:
        return (False, Direction.NONE)
    dEdx = charge_deposited / arclength

    integral_len = np.linalg.norm(cluster.data[0, :3] - vertex)
    eloss = cluster.data[0, 3]
    cutoff = 700.0  # mm
    index = 1
    while True:
        if index == len(cluster.data):
            break
        elif integral_len > cutoff:
            break
        eloss += cluster.data[index, 3]
        integral_len += np.linalg.norm(
            cluster.data[index, :3] - cluster.data[index - 1, :3]
        )
        index += 1
    cutoff_index = index
    # fill in our map
    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    results["ic_amplitude"].append(ic_amplitude)
    results["ic_centroid"].append(ic_centroid)
    results["ic_integral"].append(ic_integral)
    results["ic_multiplicity"].append(ic_multiplicity)
    results["vertex_x"].append(vertex[0])
    results["vertex_y"].append(vertex[1])
    results["vertex_z"].append(vertex[2])
    results["center_x"].append(center[0])
    results["center_y"].append(center[1])
    results["center_z"].append(center[2])
    results["polar"].append(polar)
    results["azimuthal"].append(azimuthal)
    results["brho"].append(brho)
    results["dEdx"].append(dEdx)
    results["dE"].append(charge_deposited)
    results["arclength"].append(arclength)
    results["direction"].append(direction.value)
    results["eloss"].append(eloss)
    results["cutoff_index"].append(cutoff_index)
    return (True, direction)
