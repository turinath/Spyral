from .constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE
from ..interpolate import LinearInterpolator

from spyral_utils.nuclear import NucleusData
from spyral_utils.nuclear.target import GasTarget

import math
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from pathlib import Path
import json


COARSE_TIME_WINDOW: float = 1.0e-6  # 1us window
KE_LIMIT = 0.001
DEG2RAD: float = np.pi / 180.0


@dataclass
class MeshParameters:
    """Wrapper for the general parameters required to generate a mesh of ODE solutions

    A mesh for Spyral is a 4 dimensional array of solutions to the equations of motion (also
    called track or trajectory) through the AT-TPC detector. These parameters describe
    the shape of the mesh as well as the physical properties used to generate it.

    Attributes
    ----------
    target: spyral_utils.nuclear.target.GasTarget
        The target material
    particle: spyral_utils.nuclear.NucleusData
        The projectile
    bfield: float
        The magnetic field magnitude in T
    efield: float
        The electric field magnitude in V/m
    n_time_steps: int
        The number of ODE solver timesteps
    ke_min: float
        The minimum kinetic energy of the mesh in MeV
    ke_max: float
        The maximum kinetic energy of the mesh in MeV
    ke_bins: int
        The number of kinetic energy bins in the mesh
    polar_min: float
        The minimum polar angle of the mesh in degrees
    polar_max: float
        The maximum polar angle of the mesh in degrees
    polar_bins: int
        The number of polar angle bins in the mesh

    Methods
    -------
    serialize_json() -> str
        Serialize the class to a JSON string
    """

    target: GasTarget
    particle: NucleusData
    bfield: float  # T
    efield: float  # V/m
    n_time_steps: int
    ke_min: float  # MeV
    ke_max: float  # MeV
    ke_bins: int
    polar_min: float  # deg
    polar_max: float  # deg
    polar_bins: int

    def serialize_json(self) -> str:
        """Serialize the class to a JSON string

        Returns
        -------
        str
            A JSON string
        """
        return json.dumps(
            self,
            default=lambda obj: {
                "particle": obj.particle.isotopic_symbol,
                "gas": obj.target.pretty_string,
                "bfield": obj.bfield,
                "efield": obj.efield,
                "time_steps": obj.n_time_steps,
                "ke_min": obj.ke_min,
                "ke_max": obj.ke_max,
                "ke_bins": obj.ke_bins,
                "polar_min": obj.polar_min,
                "polar_max": obj.polar_max,
                "polar_bins": obj.polar_bins,
            },
            indent=4,
        )


def check_mesh_needs_generation(track_path: Path, params: MeshParameters) -> bool:
    """Check if track mesh meta data matches or if track mesh doesn't exist

    Parameters
    ----------
    track_path: Path
        Path to the track mesh data
    params: MeshParamters
        The parameters for this track mesh

    Returns
    -------
    bool
        Returns True if the mesh needs to be generated
    """
    if track_path.exists():
        meta_path = track_path.parents[0] / f"{track_path.stem}.json"
        if not meta_path.exists():
            return True
        with open(meta_path, "r") as meta_file:
            meta_str = meta_file.read()
            return params.serialize_json() != meta_str
    else:
        return True


# State = [x, y, z, vx, vy, vz]
# Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> np.ndarray:
    """The equations of motion for a charged particle in a static electromagnetic field which experiences energy loss through some material.

    Field directions are chosen based on standard AT-TPC configuration

    Parameters
    ----------
    t: float
        time step
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field
    Efield: float
        the magnitude of the electric field
    target: Target
        the material through which the particle travels
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data

    Returns
    -------
    ndarray
        the derivatives of the state
    """

    speed = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)

    unit_vector = state[3:] / speed  # direction
    kinetic_energy = ejectile.mass * (
        1.0 / math.sqrt(1.0 - (speed / C) ** 2.0) - 1.0
    )  # MeV

    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * E_CHARGE
    qm = charge_c / mass_kg

    deceleration = (
        target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.density * 100.0
    ) / mass_kg
    results = np.zeros(6)
    results[0] = state[3]
    results[1] = state[4]
    results[2] = state[5]
    results[3] = qm * state[4] * Bfield - deceleration * unit_vector[0]
    results[4] = qm * (-1.0 * state[3] * Bfield) - deceleration * unit_vector[1]
    results[5] = qm * Efield - deceleration * unit_vector[2]

    return results


def stop_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the low-energy stopping condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data

    Returns
    -------
    float
        The difference between the kinetic energy and the lower limit. When
        this function returns zero the termination condition has been reached.

    """
    speed = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)
    kinetic_energy = ejectile.mass * (
        1.0 / math.sqrt(1.0 - (speed / C) ** 2.0) - 1.0
    )  # MeV
    return kinetic_energy - KE_LIMIT


def forward_z_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-z condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the length of the detector (1m). When
        this function returns zero the termination condition has been reached.

    """
    return state[2] - 1.0


def backward_z_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-z condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the beginning of the detector (-1.0 m). When
        this function returns zero the termination condition has been reached.

    """
    return state[2] + 1.0


def rho_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-rho condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Note here that the edge in rho (292 mm) is padded by 30 mm to account for conditions where the vertex is off axis

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the maximum rho of the detector (332 mm). When
        this function returns zero the termination condition has been reached.

    """
    return float(np.linalg.norm(state[:2])) - 0.332


def generate_track_mesh(params: MeshParameters, track_path: Path):
    """Generate a mesh of tracks given some parameters and write them to an npy file at a path.

    Creates a 4-dimensional array (mesh) of ODE solutions (also called tracks or trajectories) and
    writes them to disk. The dimensions are as follows: time x initial energy x inital polar angle x position.
    This mesh can then be used to interpolate and find an approximate solution for a given energy, polar angle.

    In many cases of AT-TPC analysis, it is diffcult to pre-estimate the time window needed to cover the full
    trajectory. To compensate, Spyral will do a first pass over the mesh using a coarse 1us time window.
    It will then estimate a finer time window based off of the results of this first pass, and then generate the
    mesh.

    Parameters
    ----------
    params: InterpolatorParameters
        parameters which control the tracks
    track_path: Path
        where to write the tracks to
    """
    kes = np.linspace(params.ke_min, params.ke_max, params.ke_bins)
    polars = np.linspace(
        params.polar_min * DEG2RAD, params.polar_max * DEG2RAD, params.polar_bins
    )

    track_meta_path = track_path.parents[0] / f"{track_path.stem}.json"
    with open(track_meta_path, "w") as metafile:
        metafile.write(params.serialize_json())

    # time x (x, y, z, vx, vy, vz) x ke x polar
    data = np.zeros(
        shape=(params.n_time_steps, 3, params.ke_bins, params.polar_bins),
        dtype=np.float64,
    )

    initial_state = np.zeros(6)

    bfield = -1.0 * params.bfield
    efield = -1.0 * params.efield

    total_iters = params.polar_bins * params.ke_bins
    count = 0
    flush_percent = 0.01
    flush_val = flush_percent * total_iters
    flush_count = 0

    longest_time = 0.0

    # Set the conditions to be terminal for a given direction (approaching from positive or negative slope)
    # See scipy docs for details
    stop_condition.terminal = True
    stop_condition.direction = -1.0
    forward_z_bound_condition.terminal = True
    forward_z_bound_condition.direction = 1.0
    backward_z_bound_condition.terminal = True
    backward_z_bound_condition.direction = -1.0
    rho_bound_condition.terminal = True
    rho_bound_condition.direction = 1.0

    print("Optimizing time range...")

    # First narrow the time range to the relevant size for the problem
    for eidx, e in enumerate(kes):
        for pidx, p in enumerate(polars):
            count += 1
            if count == flush_val:
                count = 0
                flush_count += 1
                print(
                    f"\rPercent of data generated: {int(flush_count * flush_percent * 100)}%",
                    end="",
                )

            initial_state[:] = 0.0
            gamma = e / params.particle.mass + 1.0
            speed = math.sqrt(1.0 - 1.0 / (gamma**2.0)) * C
            initial_state[3] = speed * math.sin(p)
            initial_state[5] = speed * math.cos(p)
            result = solve_ivp(
                equation_of_motion,
                (0.0, COARSE_TIME_WINDOW),
                initial_state,
                method="RK45",
                events=[
                    stop_condition,
                    forward_z_bound_condition,
                    backward_z_bound_condition,
                    rho_bound_condition,
                ],
                args=(bfield, efield, params.target, params.particle),
            )
            longest_time = max(longest_time, result.t[-1])

    print("")
    print(f"Estimated time window: {longest_time}")
    time_steps = np.geomspace(1.0e-11, longest_time, num=params.n_time_steps)
    # time_steps = np.linspace(0.0, longest_time, num=params.n_time_steps)
    flush_count = 0

    print("Generating mesh...")
    # Now redo solving, using the maximum time estimated from the first pass
    for eidx, e in enumerate(kes):
        for pidx, p in enumerate(polars):
            count += 1
            if count == flush_val:
                count = 0
                flush_count += 1
                print(
                    f"\rPercent of data generated: {int(flush_count * flush_percent * 100)}%",
                    end="",
                )

            initial_state[:] = 0.0
            gamma = e / params.particle.mass + 1.0
            speed = math.sqrt(1.0 - 1.0 / (gamma**2.0)) * C
            initial_state[3] = speed * math.sin(p)
            initial_state[5] = speed * math.cos(p)
            result = solve_ivp(
                equation_of_motion,
                (0.0, longest_time),
                initial_state,
                method="RK45",
                events=[
                    stop_condition,
                    forward_z_bound_condition,
                    backward_z_bound_condition,
                    rho_bound_condition,
                ],
                args=(bfield, efield, params.target, params.particle),
                t_eval=time_steps,
            )
            trajectory = result.y.T
            last_index = len(trajectory)
            data[:last_index, :, eidx, pidx] = trajectory[:, :3]
            if last_index < params.n_time_steps:
                # Stopped, so remaining time is just final position
                data[last_index:, :, eidx, pidx] = trajectory[-1, :3]
    print("")
    np.save(track_path, data)


def generate_interpolated_track(
    vx: float,
    vy: float,
    vz: float,
    polar: float,
    azim: float,
    ke: float,
    particle: NucleusData,
    bfield: float,
    efield: float,
    target: GasTarget,
    n_time_steps: int = 1000,
) -> LinearInterpolator | None:
    """Get a single interpolated trajectory given some initial state and system parameters

    Parameters
    -------------
    vx: float
        Vertex x-coordinate in m
    vy: float
        Vertex y-coordinate in m
    vz: float
        Vertex z-coordinate in m
    polar: float
        Polar angle in radians
    azim: float
        azimuthal angle in radians
    ke: float
        Kinetic energy in MeV
    particle: NucleusData
        The particle of interest
    bfield: float
        The magnetic field in Tesla
    efield: float
        The electric field in V/m
    target: GasTarget
        The target material
    n_time_steps: int
        The number of timesteps used by the solver
    Returns
    -------
    LinearInterpolator | None
        Returns a LinearInterpolator, which interpolates the trajectory upon z for x,y or None when the algorithm fails
    """

    # Termination conditions
    stop_condition.terminal = True
    stop_condition.direction = -1.0
    forward_z_bound_condition.terminal = True
    forward_z_bound_condition.direction = 1.0
    backward_z_bound_condition.terminal = True
    backward_z_bound_condition.direction = -1.0
    rho_bound_condition.terminal = True
    rho_bound_condition.direction = 1.0

    # Setup initial state
    gamma = ke / particle.mass + 1.0
    speed = math.sqrt(1.0 - 1.0 / (gamma**2.0)) * C
    initial_state = np.array(
        [
            vx,
            vy,
            vz,
            speed * math.sin(polar) * math.cos(azim),
            speed * math.sin(polar) * math.sin(azim),
            speed * math.cos(polar),
        ]
    )

    # Solve coarsely to deterimine problem timescale
    trajectory = solve_ivp(
        equation_of_motion,
        (0.0, COARSE_TIME_WINDOW),
        initial_state,
        method="RK45",
        events=[
            stop_condition,
            forward_z_bound_condition,
            backward_z_bound_condition,
            rho_bound_condition,
        ],
        args=(bfield, efield, target, particle),
    )
    longest_time = trajectory.t[-1]

    # Solve with refined scale to produce best results
    time_steps = np.linspace(0.0, longest_time, num=n_time_steps)
    trajectory = solve_ivp(
        equation_of_motion,
        (0.0, longest_time),
        initial_state,
        method="RK45",
        events=[
            stop_condition,
            forward_z_bound_condition,
            backward_z_bound_condition,
            rho_bound_condition,
        ],
        args=(bfield, efield, target, particle),
        t_eval=time_steps,
    )
    trajectory = trajectory.y.T
    if len(trajectory) < 2:
        return None

    # Handle data > 90 deg. LinearInterpolator requires x data (z in our case) to be monotonically increasing
    # So flip the array along axis 0 (flipud). Also trim data where sometimes particle starts going backward
    # due to efield
    if polar > np.pi * 0.5:
        trajectory = np.flipud(trajectory)
        mask = np.diff(np.ascontiguousarray(trajectory[:, 2])) > 0
        mask = np.append(mask, True)
        trajectory = trajectory[mask]

    return LinearInterpolator(trajectory[:, 2], trajectory[:, :2].T)
