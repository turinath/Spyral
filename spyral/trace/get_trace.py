from .peak import Peak
from ..core.constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS
from ..core.config import GetParameters
from ..core.hardware_id import HardwareID

from scipy import signal
import numpy as np


class GetTrace:
    """A single trace from the GET DAQ data

    Represents a raw signal from the AT-TPC pad plane through the GET data acquisition.

    Attributes
    ----------
    trace: ndarray
        The trace data
    peaks: list[Peak]
        The peaks found in the trace
    hw_id: HardwareID
        The hardware ID for the pad this trace came from

    Methods
    -------
    GetTrace(data: ndarray, id: HardwareID, params: GetParameters)
        Construct the GetTrace and find peaks
    set_trace_data(data: ndarray, id: HardwareID, params: GetParameters)
        Set the trace data and find peaks
    is_valid() -> bool:
        Check if the trace is valid
    get_pad_id() -> int
        Get the pad id for this trace
    find_peaks(params: GetParameters)
        Find the peaks in the trace
    get_number_of_peaks() -> int
        Get the number of peaks found in the trace
    get_peaks(params: GetParameters) -> list[Peak]
        Get the peaks found in the trace
    """

    def __init__(self, data: np.ndarray, id: HardwareID, params: GetParameters):
        """Construct the GetTrace and find peaks

        Parameters
        ----------
        data: ndarray
            The trace data
        id: HardwareID
            The HardwareID for the pad this trace came from
        params: GetParameters
            Configuration parameters controlling the GET signal analysis

        Returns
        -------
        GetTrace
            An instance of this class
        """
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id, params)

    def set_trace_data(self, data: np.ndarray, id: HardwareID, params: GetParameters):
        """Set trace data and find peaks

        Parameters
        ----------
        data: ndarray
            The trace data
        id: HardwareID
            The HardwareID for the pad this trace came from
        params: GetParameters
            Configuration parameters controlling the GET signal analysis
        """
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            print(
                f"GetTrace was given data that did not have the correct shape! Expected 512 time buckets, instead got {data_shape[0]}"
            )
            return

        self.trace = data.astype(np.int32)  # Widen the type and sign it
        self.hw_id = id
        self.find_peaks(params)

    def is_valid(self) -> bool:
        """Check if the trace is valid

        Returns
        -------
        bool
            If True the trace is valid
        """
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(
            self.trace, np.ndarray
        )

    def get_pad_id(self) -> int:
        """Get the pad id for this trace

        Returns
        -------
        int
            The ID number for the pad this trace came from
        """
        return self.hw_id.pad_id

    def find_peaks(self, params: GetParameters, rel_height: float = 0.95):
        """Find the peaks in the trace data

        The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
        function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.

        Parameters
        ----------
        params: GetParameters
            Configuration paramters controlling the GET signal analysis
        rel_height: float
            The relative height at which the left and right ips points are evaluated. Typically this is
            not needed to be modified, but for some legacy data is necessary
        """

        if self.is_valid() == False:
            return

        self.peaks.clear()

        pks, props = signal.find_peaks(
            self.trace,
            distance=params.peak_separation,
            prominence=params.peak_prominence,
            width=(1.0, params.peak_max_width),
            rel_height=rel_height,
        )
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props["left_ips"][idx]))
            peak.negative_inflection = int(np.ceil(props["right_ips"][idx]))
            peak.integral = np.sum(
                np.abs(self.trace[peak.positive_inflection : peak.negative_inflection])
            )
            if peak.amplitude > params.peak_threshold:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        """Get the number of peaks found in the trace

        Returns
        -------
        int
            Number of found peaks
        """
        return len(self.peaks)

    def get_peaks(self) -> list[Peak]:
        """Get the peaks found in the trace

        Returns
        -------
        list[Peak]
            The peaks found in the trace
        """
        return self.peaks
