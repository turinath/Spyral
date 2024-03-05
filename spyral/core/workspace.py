from .config import WorkspaceParameters
from .pad_map import PadMap

from spyral_utils.nuclear import NucleusData
from spyral_utils.nuclear.target import GasTarget

from pathlib import Path


def form_run_string(run_number: int) -> str:
    """Make the run_* string

    Parameters
    ----------
    run_number: int
        The run number

    Returns
    -------
    str
        The run string
    """
    return f"run_{run_number:04d}"


class Workspace:
    """The project workspace

    The Workspace class represents the disk location to which data can be written/retrieved.
    The workspace can create directories as needed for writting data. Note, that the workspace cannot
    access directories with restricted permissions.

    Attributes
    ----------
    trace_data_path: Path
        Path to merged AT-TPC data
    workspace_path: Path
        Path to the workspace location
    point_cloud_path: Path
        Path to the clouds folder in the workspace
    cluster_path: Path
        Path to the cluster folder in the workspace
    estimates_path: Path
        Path to the estimates folder in the workspace
    physics_path: Path
        Path to the physics folder in the workspace
    gate_path: Path
        Path to the gates folder in the workspace
    track_path: Path
        Path to the tracks folder in the workspace
    correction_path: Path
        Path to the correction folder in the workspace
    log_path: Path
        Path to the log folder in the workspace
    pad_geometry_path: Path
        Path to pad geometry file
    pad_gain_path: Path
        Path to pad gain file
    pad_time_path: Path
        Path to pad time correction file
    pad_electronics_path: Path
        Path to pad electronics id file
    pad_scale_path: Path
        Path to pad scale file
    self.pad_map: PadMap
        The PadMap

    Methods
    -------
    Workspace(params: WorkspaceParameters)
        Construct the workspace and make the directories
    get_trace_file_path(run_number: int) -> Path
        Get the raw trace file path for a run
    get_point_cloud_file_path(run_number: int) -> Path
        Get the point cloud file path for a run
    get_cluster_file_path(run_number: int) -> Path
        Get the cluster file path for a run
    get_estimate_file_path_parquet(run_number: int) -> Path
        Get the estimate parquet file path for a run
    get_estimate_file_path_csv(run_number: int) -> Path
        Get the estimate csv file path for a run
    get_physics_file_path_parquet(run_number: int, particle: NucleusData) -> Path
        Get the physics parquet file path for a run, particle type
    get_physics_file_path_csv(run_number: int, particle: NucleusData) -> Path
        Get the physics csv file path for a run, particle type
    get_gate_file_path(gate_name: str) -> Path
        Get the path to a gate file
    get_pad_map() -> PadMap
        Get the PadMap
    get_track_file_path(projectile: NucleusData, target: GasTarget) -> Path
        Get the track interpolation file path
    get_correction_file_path(garf_path: Path) -> Path
        Get the electron correction file path given a garfield file
    get_log_file_path(process_id: int) -> Path
        Get the log file path given a process_id
    clear_log_path() -> Path
        Clear the log directory path
    """

    def __init__(self, params: WorkspaceParameters, is_legacy: bool = False):
        """Construct the workspace and make the directories

        Parameters
        ----------
        params: WorkspaceParameters
            Configuration parameters defining the workspace

        Returns
        -------
        Workspace
            The project Workspace
        """
        self.trace_data_path = Path(params.trace_data_path)
        self.workspace_path = Path(params.workspace_path)

        if not self.workspace_path.exists():
            self.workspace_path.mkdir()

        if not self.workspace_path.is_dir() or not self.trace_data_path.is_dir():
            print(self.workspace_path)
            print(self.trace_data_path)
            raise Exception(
                "Workspace encountered an error! Trace data path and workspace path should point to directories not files!"
            )

        if not self.trace_data_path.exists():
            raise Exception(
                "Workspace encountered an error! The trace data path must exist!"
            )

        self.point_cloud_path = self.workspace_path / "clouds"
        if not self.point_cloud_path.exists():
            self.point_cloud_path.mkdir()

        self.cluster_path = self.workspace_path / "clusters"
        if not self.cluster_path.exists():
            self.cluster_path.mkdir()

        self.estimate_path = self.workspace_path / "estimates"
        if not self.estimate_path.exists():
            self.estimate_path.mkdir()

        self.physics_path = self.workspace_path / "physics"
        if not self.physics_path.exists():
            self.physics_path.mkdir()

        self.gate_path = self.workspace_path / "gates"
        if not self.gate_path.exists():
            self.gate_path.mkdir()

        self.track_path = self.workspace_path / "tracks"
        if not self.track_path.exists():
            self.track_path.mkdir()

        self.correction_path = self.workspace_path / "correction"
        if not self.correction_path.exists():
            self.correction_path.mkdir()

        self.log_path = self.workspace_path / "log"
        if not self.log_path.exists():
            self.log_path.mkdir()

        self.pad_geometry_path = Path(params.pad_geometry_path)
        if not self.pad_geometry_path.exists() or not self.pad_geometry_path.is_file():
            raise Exception(
                "Workspace encountered an error! Pad geometry path does not exist!"
            )

        self.pad_gain_path = Path(params.pad_gain_path)
        if not self.pad_gain_path.exists() or not self.pad_gain_path.is_file():
            raise Exception(
                "Workspace encountered an error! Pad gain path does not exist!"
            )

        self.pad_time_path = Path(params.pad_time_path)
        if not self.pad_time_path.exists() or not self.pad_time_path.is_file():
            raise Exception(
                "Workspace encountered an error! Pad gain path does not exist!"
            )

        self.pad_electronics_path = Path(params.pad_electronics_path)
        if (
            not self.pad_electronics_path.exists()
            or not self.pad_electronics_path.is_file()
        ):
            raise Exception(
                "Workspace encountered an error! Pad gain path does not exist!"
            )

        self.pad_scale_path = Path(__file__).parents[0] / "../../etc/pad_scale.csv"
        self.pad_map = PadMap(
            self.pad_geometry_path,
            self.pad_gain_path,
            self.pad_time_path,
            self.pad_electronics_path,
            self.pad_scale_path,
            is_legacy,
        )

    def get_trace_file_path(self, run_number: int) -> Path:
        """Get the raw trace file path for a run

        Parameters
        ----------
        run_number: int
            The run number

        Returns
        -------
        Path
            The trace file path
        """
        runstr = form_run_string(run_number)
        return self.trace_data_path / f"{runstr}.h5"

    def get_point_cloud_file_path(self, run_number: int) -> Path:
        """Get the point cloud file path for a run

        Parameters
        ----------
        run_number: int
            The run number

        Returns
        -------
        Path
            The point cloud file path
        """
        runstr = form_run_string(run_number)
        return self.point_cloud_path / f"{runstr}.h5"

    def get_cluster_file_path(self, run_number: int) -> Path:
        """Get the cluster file path for a run

        Parameters
        ----------
        run_number: int
            The run number

        Returns
        -------
        Path
            The cluster file path
        """
        runstr = form_run_string(run_number)
        return self.cluster_path / f"{runstr}.h5"

    def get_estimate_file_path_parquet(self, run_number: int) -> Path:
        """Get the estimate parquet file path for a run

        Parameters
        ----------
        run_number: int
            The run number

        Returns
        -------
        Path
            The estimate parquet file path
        """
        runstr = form_run_string(run_number)
        return self.estimate_path / f"{runstr}.parquet"

    def get_estimate_file_path_csv(self, run_number: int) -> Path:
        """Get the estimate csv file path for a run

        Parameters
        ----------
        run_number: int
            The run number

        Returns
        -------
        Path
            The estimate csv file path
        """
        runstr = form_run_string(run_number)
        return self.estimate_path / f"{runstr}.csv"

    def get_physics_file_path_parquet(
        self, run_number: int, particle: NucleusData
    ) -> Path:
        """Get the phsyics parquet file path for a run

        Parameters
        ----------
        run_number: int
            The run number
        particle: NucleusData
            The particle that was used for the solve

        Returns
        -------
        Path
            The physics parquet file path
        """
        runstr = form_run_string(run_number)
        return self.physics_path / f"{runstr}_{particle.isotopic_symbol}.parquet"

    def get_physics_file_path_csv(self, run_number: int, particle: NucleusData) -> Path:
        """Get the phsyics csv file path for a run

        Parameters
        ----------
        run_number: int
            The run number
        particle: NucleusData
            The particle that was used for the solve

        Returns
        -------
        Path
            The physics csv file path
        """
        runstr = form_run_string(run_number)
        return self.physics_path / f"{runstr}_{particle.isotopic_symbol}.csv"

    def get_gate_file_path(self, gate_name: str) -> Path:
        """Get the gate file path

        Parameters
        ----------
        gate_name: str
            The gate file name

        Returns
        -------
        Path
            The gate file path
        """
        return self.gate_path / gate_name

    def get_pad_map(self) -> PadMap:
        return self.pad_map

    def get_track_file_path(self, projectile: NucleusData, target: GasTarget) -> Path:
        """Get the track interpolation scheme file path

        Parameters
        ----------
        projectile: NucleusData
            The projectile data
        target: GasTarget
            The target material

        Returns
        -------
        Path
            The track interpolation scheme file path
        """
        return (
            self.track_path
            / f"{projectile.isotopic_symbol}_in_{target.ugly_string.replace('(Gas)', '')}_{target.data.pressure}Torr.npy"
        )

    def get_correction_file_path(self, garf_path: Path) -> Path:
        """Get the electron correction file path

        Parameters
        ----------
        garf_path: Path
            The garfield file path

        Returns
        -------
        Path
            The electron correction path
        """
        return self.correction_path / f"{garf_path.stem}.npy"

    def get_log_file_path(self, process_id: int) -> Path:
        """Get the log file path given a process id

        Parameters
        ----------
        process_id: int
            The process id. If -1, this is the parent process

        Returns
        -------
        Path
            The log file path
        """
        if process_id != -1:
            return self.log_path / f"log_proc{process_id}.txt"
        else:
            return self.log_path / "log_procParent.txt"

    def clear_log_path(self):
        """Clear the log directory"""
        for item in self.log_path.iterdir():
            if item.is_file():
                item.unlink()
