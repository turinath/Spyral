from ..core.constants import DEG2RAD
from .bilinear import BilinearInterpolator
from .linear import LinearInterpolator

import numpy as np
from pathlib import Path
import json

from numba import float64, int32
from numba.types import string, ListType
from numba.typed import List
from numba.experimental import jitclass


# To use numba with a class we need to declare the types of all members of the class
# and use the @jitclass decorator
tinterp_spec = [
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


@jitclass(spec=tinterp_spec)
class TrackInterpolator:
    """Represents an interpolation scheme used to generate trajectories.

    Solving ODE's can be expensive, so to save time pre-generate a range of solutions (mesh) and then
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
    interpolators: ListType[BilinearInterpolator]
        A list of BilinearInterpolators, one for each time step in the mesh

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
        Check if the given ke, polar point is within the mesh
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

    def get_trajectory(
        self, vx: float, vy: float, vz: float, polar: float, azim: float, ke: float
    ) -> np.ndarray | None:
        """Get a trajectory given some initial state.

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
        ndarray | None
            Returns a Nx3 ndarray of the trajectory data or None when the algorithm fails
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

        return trajectory

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
    """Create a TrackInterpolator, loading a mesh of trajectories from disk.

    This is a utility function wrapping the creation of a TrackInterpolator. We do this outside
    of the jitclass as I/O seems to only be somewhat supported in numba.

    Parameters
    ----------
    track_path: Path
        Path to the track mesh data

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
