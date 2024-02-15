from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Any


@dataclass
class WorkspaceParameters:
    """Parameters describing the core paths used by the Workspace

    Attributes
    ----------
    trace_data_path: str
        Path to the hdf5 files containing merged AT-TPC traces
    workspace_path: str
        Path to the project Workspace
    pad_geometry_path: str
        Path to the csv file containing the pad geometry
    pad_gain_path: str
        Path to the csv file containing the relative pad gains
    pad_time_path: str
        Path to the csv file containing the pad time corrections
    pad_electronics_path: str
        Path to the csv file containing the pad electronics ids
    """

    trace_data_path: str = ""
    workspace_path: str = ""
    pad_geometry_path: str = ""
    pad_gain_path: str = ""
    pad_time_path: str = ""
    pad_electronics_path: str = ""


@dataclass
class RunParameters:
    """Parameters describing the set of operations to be peformed by Spyral

    Attributes
    ----------
    run_min: int
        The first run number, inclusive
    run_max: int
        The last run number, inclusive
    n_processes: int
        The number of child processes
    do_pointcloud: bool
        If true, data is run through the pointcloud phase
    do_cluster: bool
        If true, data is run through the cluster phase
    do_estimate: bool
        If true, data is run through the estimate phase
    do_solve: bool
        If true, data is run through the solve phase
    is_legacy: bool
        If true, data is considered AT-TPC legacy format,
        before the introduction of the FRIBDAQ.
    """

    run_min: int = -1
    run_max: int = -1
    n_processes: int = -1
    do_pointcloud: bool = False
    do_cluster: bool = False
    do_estimate: bool = False
    do_solve: bool = False
    is_legacy: bool = False


@dataclass
class DetectorParameters:
    """Parameters describing the detector configuration

    Attributes
    ----------
    magnetic_field: float
        The magnitude of the magnetic field in Tesla
    electric_field: float
        The magnitude of the electric field in V/m
    detector_length: float
        The detector length in mm
    beam_region_radius: float
        The beam region radius in mm
    micromegas_time_bucket: float
        The micromegas time reference in GET time buckets
    window_time_bucket: float
        The window time reference in GET time buckets
    get_frequency: float
        The GET DAQ sampling frequency in MHz. Typically 3.125 or 6.25
    garfield_file_path: str
        Path to a Garfield simulation file containing electron drift corrections

    """

    magnetic_field: float = 0.0  # Tesla
    electric_field: float = 0.0  # V/m
    detector_length: float = 0.0  # mm
    beam_region_radius: float = 0.0  # mm
    micromegas_time_bucket: float = 0.0
    window_time_bucket: float = 0.0
    get_frequency: float = 0.0  # MHz
    garfield_file_path: str = ""
    do_garfield_correction: bool = False


@dataclass
class GetParameters:
    """Parameters for GET trace signal analysis

    Attributes
    ----------
    baseline_window_scale: float
        The scale factor for the basline correction algorithm
    peak_separation: float
        The peak separation parameter used in scipy.signal.find_peaks
    peak_prominence: float
        The peak prominence parameter used in scipy.signal.find_peaks
    peak_max_width: float
        The maximum peak width parameter used in scipy.signal.find_peaks
    peak_threshold: float
        The minimum amplitude of a valid peak
    do_sat_fit: bool
        If true, fits saturated peaks
    saturation_threshold: float
        The threshold above which a trace is flagged as saturated.
    """

    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0
    do_sat_fit: bool = False
    saturation_threshold: float = 4090.0


@dataclass
class FribParameters:
    """Parameters for FRIBDAQ (IC, Si, etc) trace signal analysis

    Attributes
    ----------
    baseline_window_scale: float
        The scale factor for the basline correction algorithm
    peak_separation: float
        The peak separation parameter used in scipy.signal.find_peaks
    peak_prominence: float
        The peak prominence parameter used in scipy.signal.find_peaks
    peak_max_width: float
        The maximum peak width parameter used in scipy.signal.find_peaks
    peak_threshold: float
        The minimum amplitude of a valid peak
    ic_multiplicity: int
        The maximum allowed ion chamber multiplicity
    correct_ic_time: bool
        If true, the ion chamber time correction is used to correct the GET trace time
    """

    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0
    ic_multiplicity: int = 1
    correct_ic_time: bool = True


@dataclass
class ClusterParameters:
    """Parameters for clustering, cluster joining, and cluster cleaning

    Attributes
    ----------
    min_cloud_size: int
        The minimum size for a point cloud to be clustered
    smoothing_neighbor_distance: float
        Size of neighborhood radius in mm for smoothing
    min_points: int
        min_samples parameter in scikit-learns' HDBSCAN algorithm
    big_event_cutoff: int
        the cutoff between big events and small events in units of points in the
        point cloud
    min_size_scale_factor: int
        Factor which is multiplied by the number of points in a point cloud to set
        the min_cluster_size parameter in scikit-learn's HDBSCAN algorithm
    min_size_lower_cutoff: int
        min_cluster_size parameter in scikit-learn's HDBSCAN algorithm for events where n_points * min_size_scale_factor
        are less than this value.
    circle_overlap_ratio: float
        minimum overlap ratio between two circles in the cluster joining algorithm
    fractional_charge_threshold: float
        The maximum allowed difference between two clusters mean charge (relative to the larger mean charge of the two)
        for them to be joined
    n_neighbors_outlier_test: int
        Number of neighbors to use in scikit-learn's LocalOutlierFactor test
    """

    min_cloud_size: int = 0
    smoothing_neighbor_distance: float = 0.0  # mm
    min_points: int = 0
    min_size_scale_factor: float = 0.0
    min_size_lower_cutoff: int = 0
    circle_overlap_ratio: float = 0.0
    fractional_charge_threshold: float = 0.0
    n_neighbors_outlier_test: int = 0


@dataclass
class EstimateParameters:
    """Parameters for physics estimation

    Attributes
    ----------
    min_total_trajectory_points: int
        minimum number of points in a cluster for the cluster to be considered a particle trajectory
    max_distance_from_beam_axis: float
        maximum distance from beam axis for a trajectory vertex to be considered valid
    """

    min_total_trajectory_points: int = 0
    max_distance_from_beam_axis: float = 0.0  # mm


@dataclass
class SolverParameters:
    """Parameters for physics solving

    Attributes
    ----------
    gas_data_path: str
        Path to a spyral-utils GasTarget file
    particle_id_filename: str
        Name of a particle ID cut file
    ic_min_val: float
        Low value the desired beam region of the ion chamber spectrum
    ic_max_value: float
        High value the desired beam region of the ion chamber spectrum
    n_time_steps: int
        The number of timesteps used in the ODE solver
    interp_ke_min: float
        The minimum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_max: float
        The maximum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_bins: int
        The number of kinetic energy bins used in the interpolation scheme
    interp_polar_min: float
        The minimum value of polar angle used in the interpolation scheme in degrees
    interp_polar_max: float
        The maximum value of polar angle used in the interpolation scheme in degrees
    interp_polar_bins: int
        The number of polar angle bins used in the interpolation scheme
    """

    gas_data_path: str = ""
    particle_id_filename: str = ""
    ic_min_val: float = 0.0
    ic_max_val: float = 0.0
    n_time_steps: int = 0
    interp_ke_min: float = 0.0
    interp_ke_max: float = 0.0
    interp_ke_bins: int = 0
    interp_polar_min: float = 0.0
    interp_polar_max: float = 0.0
    interp_polar_bins: int = 0


@dataclass
class Config:
    """Container which holds all configuration parameters.

    Can be serialized/deserialized to json.

    Attributes
    ----------
    workspace: WorkspaceParameters
    run: RunParameters
    detector: DetectorParameters
    get: GetParameters
    frib: FribParameters
    cluster: ClusterParameters
    estimate: EstimateParameters
    solver: SolverParameters
    """

    # Workspace
    workspace: WorkspaceParameters = field(default_factory=WorkspaceParameters)

    # Run
    run: RunParameters = field(default_factory=RunParameters)

    # Detector
    detector: DetectorParameters = field(default_factory=DetectorParameters)

    # Traces
    get: GetParameters = field(default_factory=GetParameters)

    # FRIB data
    frib: FribParameters = field(default_factory=FribParameters)

    # Clustering settings
    cluster: ClusterParameters = field(default_factory=ClusterParameters)

    # Physics Estimate settings
    estimate: EstimateParameters = field(default_factory=EstimateParameters)

    # Physics Solver
    solver: SolverParameters = field(default_factory=SolverParameters)


def deserialize_config(json_data: dict[Any, Any]) -> Config:
    """Deserialize the Config from some json_data dictionary

    Parameters
    ----------
    json_data: dict[Any, Any]
        Dictionary made by a json.load operation

    Returns
    -------
    Config
        The deserialized Config
    """
    config = Config()
    ws_params = json_data["Workspace"]
    config.workspace.trace_data_path = ws_params["trace_data_path"]
    config.workspace.workspace_path = ws_params["workspace_path"]
    config.workspace.pad_geometry_path = ws_params["pad_geometry_path"]
    config.workspace.pad_gain_path = ws_params["pad_gain_path"]
    config.workspace.pad_time_path = ws_params["pad_time_path"]
    config.workspace.pad_electronics_path = ws_params["pad_electronics_path"]

    run_params = json_data["Run"]
    config.run.run_min = run_params["run_min"]
    config.run.run_max = run_params["run_max"]
    config.run.n_processes = run_params["n_processes"]
    config.run.do_pointcloud = run_params["phase_pointcloud"]
    config.run.do_cluster = run_params["phase_cluster"]
    config.run.do_estimate = run_params["phase_estimate"]
    config.run.do_solve = run_params["phase_solve"]
    config.run.is_legacy = run_params["is_legacy"]

    det_params = json_data["Detector"]
    config.detector.magnetic_field = det_params["magnetic_field(T)"]
    config.detector.electric_field = det_params["electric_field(V/m)"]
    config.detector.detector_length = det_params["detector_length(mm)"]
    config.detector.beam_region_radius = det_params["beam_region_radius(mm)"]
    config.detector.micromegas_time_bucket = det_params["micromegas_time_bucket"]
    config.detector.window_time_bucket = det_params["window_time_bucket"]
    config.detector.get_frequency = det_params["get_frequency(MHz)"]
    config.detector.garfield_file_path = det_params["electric_field_garfield_path"]
    config.detector.do_garfield_correction = det_params["do_garfield_correction"]

    get_params = json_data["GET"]
    config.get.baseline_window_scale = get_params["baseline_window_scale"]
    config.get.peak_separation = get_params["peak_separation"]
    config.get.peak_prominence = get_params["peak_prominence"]
    config.get.peak_max_width = get_params["peak_max_width"]
    config.get.peak_threshold = get_params["peak_threshold"]
    config.get.do_sat_fit = get_params["do_sat_fit"]
    config.get.saturation_threshold = get_params["saturation_threshold"]

    frib_params = json_data["FRIB"]
    config.frib.baseline_window_scale = frib_params["baseline_window_scale"]
    config.frib.peak_separation = frib_params["peak_separation"]
    config.frib.peak_prominence = frib_params["peak_prominence"]
    config.frib.peak_max_width = frib_params["peak_max_width"]
    config.frib.peak_threshold = frib_params["peak_threshold"]
    config.frib.ic_multiplicity = frib_params["event_ic_multiplicity"]
    config.frib.correct_ic_time = frib_params["event_correct_ic_time"]

    cluster_params = json_data["Cluster"]
    config.cluster.min_cloud_size = cluster_params["min_cloud_size"]
    config.cluster.smoothing_neighbor_distance = cluster_params[
        "smoothing_neighbor_distance(mm)"
    ]
    config.cluster.min_size_scale_factor = cluster_params["minimum_size_scale_factor"]
    config.cluster.min_size_lower_cutoff = cluster_params["minimum_size_lower_cutoff"]
    config.cluster.min_points = cluster_params["minimum_points"]
    config.cluster.circle_overlap_ratio = cluster_params["circle_overlap_ratio"]
    config.cluster.fractional_charge_threshold = cluster_params[
        "fractional_charge_threshold"
    ]
    config.cluster.n_neighbors_outlier_test = cluster_params["n_neighbors_outlier_test"]

    est_params = json_data["Estimate"]
    config.estimate.min_total_trajectory_points = est_params[
        "mininum_total_trajectory_points"
    ]
    config.estimate.max_distance_from_beam_axis = est_params[
        "maximum_distance_from_beam_axis"
    ]

    solver_params = json_data["Solver"]
    config.solver.gas_data_path = solver_params["gas_data_path"]
    config.solver.particle_id_filename = solver_params["particle_id_file"]
    config.solver.ic_min_val = solver_params["ic_min"]
    config.solver.ic_max_val = solver_params["ic_max"]
    config.solver.n_time_steps = solver_params["ode_n_time_steps"]
    config.solver.interp_ke_min = solver_params["interp_ke_min(MeV)"]
    config.solver.interp_ke_max = solver_params["interp_ke_max(MeV)"]
    config.solver.interp_ke_bins = solver_params["interp_ke_bins"]
    config.solver.interp_polar_min = solver_params["interp_polar_min(deg)"]
    config.solver.interp_polar_max = solver_params["interp_polar_max(deg)"]
    config.solver.interp_polar_bins = solver_params["interp_polar_bins"]

    return config


def load_config(file_path: Path) -> Config:
    """Load a configuration from a file

    Parameters
    ----------
    file_path: Path
        Path to a JSON file containing a Spyral configuration

    Returns
    -------
    Config:
        The Spyral configuration
    """
    with open(file_path, "r") as json_file:
        config_data = load(json_file)
        return deserialize_config(config_data)
