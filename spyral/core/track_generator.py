from .constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE
from ..interpolate import BilinearInterpolator, LinearInterpolator

from spyral_utils.nuclear import NucleusData
from spyral_utils.nuclear.target import GasTarget

import math
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from pathlib import Path
import json

from numba import float64, int32, njit
from numba.types import string, ListType
from numba.typed import List
from numba.experimental import jitclass

TIME_WINDOW: float = 1.0e-6  # 1us window
KE_LIMIT = 0.1
DEG2RAD: float = np.pi / 180.0


@njit
def is_float_equal(lhs: float, rhs: float) -> bool:
    """JIT-ed is equal with absolute precision 1.0e-6

    Parameters
    ----------
    lhs: float
        The left-hand side of the comparsion
    rhs: float
        The right-hand side of the comparsion

    Returns
    -------
    bool:
        Returns True if the lhs and rhs are equal within 1.0e-6 precision

    """
    PRECISION: float = 1.0e-6
    return abs(lhs - rhs) < PRECISION


@dataclass
class InitialState:
    """Wrapper for the initial state used to generate a trajectory

    Attributes
    ---------
    vertex_x: float
        Vertex x-position in meters
    vertex_y: float
        Vertex y-position in meters
    vertex_z: float
        Vertex z-position in meters
    polar: float
        Polar angle in radians
    azimuthal: float
        Azimuthal angle in radians
    kinetic_energy: float
        Kinetic energy in MeV

    Methods
    -------
    to_array() -> ndarray
        Convert the class to an array (element order preserved)
    """

    vertex_x: float = 0.0  # m
    vertex_y: float = 0.0
    vertex_z: float = 0.0
    polar: float = 0.0  # rad
    azimuthal: float = 0.0
    kinetic_energy: float = 0.0  # MeV

    def to_array(self) -> np.ndarray:
        """Convert the class to an array (element order preserved)

        Returns
        -------
        ndarray
            Length 6 array of [x,y,z,polar,azim,ke]
        """
        return np.array(
            [
                self.vertex_x,
                self.vertex_y,
                self.vertex_z,
                self.polar,
                self.azimuthal,
                self.kinetic_energy,
            ]
        )


@dataclass
class GeneratorParams:
    """Wrapper for the general parameters required to generate an interpolation scheme

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
        The minimum kinetic energy of the interpolation scheme in MeV
    ke_max: float
        The maximum kinetic energy of the interpolation scheme in MeV
    ke_bins: int
        The number of kinetic energy bins in the interpolation scheme
    polar_min: float
        The minimum polar angle of the interpolation scheme in degrees
    polar_max: float
        The maximum polar angle of the interpolation scheme in degrees
    polar_bins: int
        The number of polar angle bins in the interpolation scheme

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
    if kinetic_energy < KE_LIMIT:
        return np.array([state[0], state[1], state[2], 0.0, 0.0, 0.0])
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


def check_tracks_need_generation(track_path: Path, params: GeneratorParams) -> bool:
    """Check if track meta data matches or if tracks don't exist

    Parameters
    ----------
    track_path: Path
        Path to the track interpolation scheme
    params: GeneratorParams
        The parameters for this interpolation scheme

    Returns
    -------
    bool
        Returns True if the interpolation scheme needs to be generated
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


def generate_tracks(params: GeneratorParams, track_path: Path):
    """Generate a set of tracks given some parameters and write them to an hdf5 file at a path.

    Parameters
    ----------
    params: GeneratorParams
        parameters which control the tracks
    track_path: Path
        where to write the tracks to
    """
    kes = np.linspace(params.ke_min, params.ke_max, params.ke_bins)
    polars = np.linspace(
        params.polar_min * DEG2RAD, params.polar_max * DEG2RAD, params.polar_bins
    )
    time_steps = np.linspace(0.0, TIME_WINDOW, num=params.n_time_steps)

    track_meta_path = track_path.parents[0] / f"{track_path.stem}.json"
    with open(track_meta_path, "w") as metafile:
        metafile.write(params.serialize_json())

    data = np.zeros(
        shape=(params.n_time_steps, 3, params.ke_bins, params.polar_bins),
        dtype=np.float64,
    )  # time x (x, y, z, vx, vy, vz) x ke x polar

    initial_state = np.zeros(6)

    bfield = -1.0 * params.bfield
    efield = -1.0 * params.efield

    total_iters = params.polar_bins * params.ke_bins
    count = 0
    flush_percent = 0.01
    flush_val = flush_percent * total_iters
    flush_count = 0

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
            momentum = math.sqrt(e * (e + 2.0 * params.particle.mass))
            speed = momentum / params.particle.mass * C
            initial_state[3] = speed * math.sin(p)
            initial_state[5] = speed * math.cos(p)
            result = solve_ivp(
                equation_of_motion,
                (0.0, TIME_WINDOW),
                initial_state,
                method="RK45",
                args=(bfield, efield, params.target, params.particle),
                t_eval=time_steps,
            )
            data[:, :, eidx, pidx] = result.y.T[:, :3]

    print("\n")

    np.save(track_path, data)


# To use numba with a class we need to declare the types of all members of the class
# and use the @jitclass decorator
@jitclass(
    [
        ("filepath", string),
        ("particle_name", string),
        ("gas_name", string),
        ("bfield", float64),
        ("efield", float64),
        ("ke_min", float64),
        ("ke_max", float64),
        ("ke_bins", int32),
        ("polar_min", float64),
        ("polar_max", float64),
        ("polar_bins", int32),
        ("interpolators", ListType(BilinearInterpolator.class_type.instance_type)),
    ]
)
class TrackInterpolator:
    """Represents an interpolation scheme used to generate trajectories.

    Solving ODE's can be expensive, so to save time pre-generate a range of solutions and then
    interpolate on these solutions. TrackInterpolator uses bilinear interpolation to
    interpolate on the energy and polar angle (reaction angle) of the trajectory.

    We use numba to just-in-time compile these methods, which results in a dramatic speed up on the order
    of a factor of 50.

    Attriubtes
    ----------
    file_path: str
        The track save file
    particle_name: str
        The isotopic symbol of the ejectile
    gas_name: str
        The target gas name
    bfield: float
        The magnetic field magnitude in T
    efield: float
        The electric field magnitude in V/m
    ke_min: float
        The minimum kinetic energy of the interpolation scheme in MeV
    ke_max: float
        The maximum kinetic energy of the interpolation scheme in MeV
    ke_bins: int
        The number of kinetic energy bins in the interpolation scheme
    polar_min: float
      The minimum polar angle of the interpolation scheme in degrees
    polar_max: float
      The maximum polar angle of the interpolation scheme in degrees
    polar_bins: int
        The number of polar angle bins in the interpolation scheme
    interpolators: ListType[BilinearInterpolator]
        A list of BilinearInterpolators, one for each time step

    Methods
    -------
    TrackInterpolator(
        track_path: str,
        interpolators: ListType[BilinearInterpolator],
        particle_name: str,
        gas_name: str,
        bfield: float,
        efield: float,
        ke_min: float,
        ke_max: float,
        ke_bins: int,
        polar_min: float,
        polar_max: float,
        polar_bins: int)
        Construct a TrackInterpolator

    get_interpolated_trajectory(vx: float, vy: float, vz: float, polar: float, azim: float, ke: float) -> LinearInterpolator | None
        Given some initial state, get an interpolated trajectory

    check_interpolator(
        particle: str,
        bfield: float,
        efield: float,
        target: str,
        ke_min: float,
        ke_max: float,
        ke_bins: int,
        polar_min: float,
        polar_max: float,
        polar_bins: int,
    ) -> bool
        Check if this interpolator matches the given values

    check_values_in_range(ke: float, polar: float) -> bool
        Check if the given ke, polar point is within the interpolation scheme
    """

    def __init__(
        self,
        track_path: str,
        interpolators: ListType(BilinearInterpolator.class_type.instance_type),
        particle_name: str,
        gas_name: str,
        bfield: float,
        efield: float,
        ke_min: float,
        ke_max: float,
        ke_bins: int,
        polar_min: float,
        polar_max: float,
        polar_bins: int,
    ):
        """Construct a TrackInterpolator

        Parameters
        ----------
        track_path: str
            Path to an interpolation file
        interpolators: ListType[BilinearInterpolator]
            A set of BilinearInterpolators
        particle_name: str
            The isotopic symbol of the particle
        gas_name: str
            The gas target name
        bfield: float
            The magnetic field magnitude in T
        efield: float
            The electric field magnitude in V/m
        ke_min: float
            The minimum kinetic energy of the interpolation scheme in MeV
        ke_max: float
            The maximum kinetic energy of the interpolation scheme in MeV
        ke_bins: int
            The number of kinetic energy bins in the interpolation scheme
        polar_min: float
            The minimum polar angle of the interpolation scheme in degrees
        polar_max: float
            The maximum polar angle of the interpolation scheme in degrees
        polar_bins: int
            The number of polar angle bins in the interpolation scheme

        Returns
        -------
        TrackInterpolator
            An instance of the class
        """
        self.filepath = track_path
        self.particle_name: str = particle_name
        self.gas_name: str = gas_name
        self.bfield: float = bfield
        self.efield: float = efield
        self.ke_min: float = ke_min
        self.ke_max: float = ke_max
        self.ke_bins: int = ke_bins
        self.polar_min: float = polar_min
        self.polar_max: float = polar_max
        self.polar_bins: int = polar_bins

        self.interpolators = interpolators

    def get_interpolated_trajectory(
        self, vx: float, vy: float, vz: float, polar: float, azim: float, ke: float
    ) -> LinearInterpolator | None:
        """Get an interpolated trajectory given some initial state.

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

        Returns
        -------
        LinearInterpolator | None
            Returns a LinearInterpolator, which interpolates the trajectory upon z for x,y or None when the algorithm fails
        """

        trajectory = np.zeros((len(self.interpolators), 3))
        for idx, _ in enumerate(trajectory):
            trajectory[idx] = self.interpolators[idx].interpolate(polar, ke)

        # Rotate the trajectory in azimuthal (around z) to match data
        z_rot = np.array(
            [
                [np.cos(azim), -np.sin(azim), 0.0],
                [np.sin(azim), np.cos(azim), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        trajectory = (z_rot @ trajectory.T).T
        # Translate to vertex
        trajectory[:, 0] += vx
        trajectory[:, 1] += vy
        trajectory[:, 2] += vz
        # Rotate
        # Trim stopped region
        removal = np.full(len(trajectory), True)
        previous_element = np.full(3, -1.0)
        for idx, element in enumerate(trajectory):
            if np.all(previous_element[:] == element[:]):
                removal[idx] = False
            previous_element = element

        trajectory = trajectory[removal]
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

    def check_interpolator(
        self,
        particle: str,
        bfield: float,
        efield: float,
        target: str,
        ke_min: float,
        ke_max: float,
        ke_bins: int,
        polar_min: float,
        polar_max: float,
        polar_bins: int,
    ) -> bool:
        """Check to see if this interpolator matches the given parameters

        Parameters
        ----------
        particle: str
            The isotopic symbol of the particle
        bfield: float
            The magnetic field magnitude in T
        efield: float
            The electric field magnitude in V/m
        target: str
            The gas target name
        ke_min: float
            The minimum kinetic energy of the interpolation scheme in MeV
        ke_max: float
            The maximum kinetic energy of the interpolation scheme in MeV
        ke_bins: int
            The number of kinetic energy bins in the interpolation scheme
        polar_min: float
            The minimum polar angle of the interpolation scheme in degrees
        polar_max: float
            The maximum polar angle of the interpolation scheme in degrees
        polar_bins: int
            The number of polar angle bins in the interpolation scheme

        Returns
        -------
        bool
            Returns true if the interpolator matches
        """
        is_valid = (
            particle == self.particle_name
            and bfield == self.bfield
            and efield == self.efield
            and target == self.gas_name
            and ke_min == self.ke_min
            and ke_max == self.ke_max
            and ke_bins == self.ke_bins
            and polar_min == self.polar_min
            and polar_max == self.polar_max
            and polar_bins == self.polar_bins
        )

        if is_valid:
            return True
        else:
            return False

    def check_values_in_range(self, ke: float, polar: float) -> bool:
        """Check if these values of energy, angle are within the interpolation range

        Parameters
        ----------
        ke: float
            The kinetic energy to check in MeV
        polar: float
            The polar angle to check in radians

        Returns
        -------
        bool
            Returns true if they are within the interpolation range
        """
        polar_deg = polar / DEG2RAD
        if (
            ke > self.ke_max
            or ke < self.ke_min
            or polar_deg < self.polar_min
            or polar_deg > self.polar_max
        ):
            return False
        else:
            return True


def create_interpolator(track_path: Path) -> TrackInterpolator:
    """Create a TrackInterpolator

    This is a utility function wrapping the creation of a TrackInterpolator. Numba doesn't support h5py for
    obvious reasons, so we need to retrieve the track data outside of the class and then manually pass it all
    to the initialization.

    Parameters
    ----------
    track_path: Path
        Path to the track interpolation data

    Returns
    -------
    TrackInterpolator
        The constructed interpolator
    """
    track_meta_path = track_path.parents[0] / f"{track_path.stem}.json"
    meta_dict: dict
    with open(track_meta_path, "r") as metafile:
        meta_dict = json.load(metafile)
    data = np.load(track_path)

    pmin_rad = meta_dict["polar_min"] * DEG2RAD
    pmax_rad = meta_dict["polar_max"] * DEG2RAD

    typed_interpolators = List()

    [
        typed_interpolators.append(
            BilinearInterpolator(
                pmin_rad,
                pmax_rad,
                meta_dict["polar_bins"],
                meta_dict["ke_min"],
                meta_dict["ke_max"],
                meta_dict["ke_bins"],
                time.T[:, :, :3],
            )
        )
        for time in data
    ]

    return TrackInterpolator(
        str(track_path),
        typed_interpolators,
        meta_dict["particle"],
        meta_dict["gas"],
        meta_dict["bfield"],
        meta_dict["efield"],
        meta_dict["ke_min"],
        meta_dict["ke_max"],
        meta_dict["ke_bins"],
        meta_dict["polar_min"],
        meta_dict["polar_max"],
        meta_dict["polar_bins"],
    )
