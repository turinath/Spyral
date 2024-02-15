from .get_trace import GetTrace
from ..core.config import GetParameters
from ..core.constants import INVALID_EVENT_NAME, INVALID_EVENT_NUMBER
from ..core.hardware_id import hardware_id_from_array

import numpy as np
import h5py as h5
from math import erf
from scipy.optimize import curve_fit
from numba import njit

GET_DATA_TRACE_START: int = 5
GET_DATA_TRACE_STOP: int = 512 + 5


class GetEvent:
    """Class representing an event in the GET DAQ

    Contains traces (GetTraces) from the AT-TPC pad plane.

    Attributes
    ----------
    traces: list[GetTrace]
        The pad plane traces from the event
    name: str
        The event name
    number:
        The event number

    Methods
    -------
    GetEvent(raw_data: h5py.Dataset, event_number: int, params: GetParameters)
        Construct the event and process traces
    load_traces(raw_data: h5py.Dataset, event_number: int, params: GetParameters)
        Process traces
    is_valid() -> bool
        Check if the event is valid
    """

    def __init__(self, raw_data: h5.Dataset, event_number: int, params: GetParameters):
        """Construct the event and process traces

        Parameters
        ----------
        raw_data: h5py.Dataset
            The hdf5 Dataset that contains trace data
        event_number: int
            The event number
        params: GetParameters
            Configuration parameters controlling the GET signal analysis

        Returns
        -------
        GetEvent
            An instance of the class
        """
        self.traces: list[GetTrace] = []
        self.raw_traces: list[GetTrace] = []
        self.name: str = INVALID_EVENT_NAME
        self.number: int = INVALID_EVENT_NUMBER
        self.load_traces(raw_data, event_number, params)

    def load_traces(
        self, raw_data: h5.Dataset, event_number: int, params: GetParameters
    ):
        """Process the traces

        Parameters
        ----------
        raw_data: h5py.Dataset
            The hdf5 Dataset that contains trace data
        event_number: int
            The event number
        params: GetParameters
            Configuration parameters controlling the GET signal analysis
        """
        self.name = str(raw_data.name)
        self.number = event_number
        trace_matrix = preprocess_traces(
            raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
            params.baseline_window_scale,
        )
        self.traces = [
            GetTrace(trace_matrix[idx], hardware_id_from_array(row[0:5]), params)
            for idx, row in enumerate(raw_data)
        ]

        if params.do_sat_fit == True:
            self.raw_traces = [
                GetTrace(
                    raw_data[idx, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
                    hardware_id_from_array(row[0:5]),
                    params,
                )
                for idx, row in enumerate(raw_data)
            ]
            self.traces = fix_sat_peaks(
                self.raw_traces.copy(),
                self.traces.copy(),
                params.saturation_threshold,
                params,
            )

    def is_valid(self) -> bool:
        return self.name != INVALID_EVENT_NAME and self.number != INVALID_EVENT_NUMBER

# Define fit function
@njit
def SkewedGaussian(x, A, sigma, alpha, shift):
    """JIT-ed functional form for a skewed Gaussian which will be fitted in fix_sat_peaks.

    Parameters
    ----------
    x: ndarray
        x-data for the fit. In our case this is the time bucket range
    A: float
        Amplitude for the Gaussian
    sigma: float
        Standard deviation for the Gaussian
    alpha: float
        Amount of "skewed-ness" of the Gaussian
    shift: float 
        Centroid of the Gaussian
    Returns
    -------
    ndarray
        Values of the curve evaluated over the x-grid with the given parameters
    """
    return (
        A
        * np.exp(-((x - shift) ** 2) / 2 / sigma ** 2)
        * (1 + np.array([erf(alpha * (x_i-shift) / sigma) for x_i in x]))
    )

# Function for fixing saturated peaks
def fix_sat_peaks(
    raw_traces: list[GetTrace],
    adj_traces: list[GetTrace],
    sat_thresh: float,
    params: GetParameters,
):
    """Method for fitting saturated traces

    Parameters
    ----------
    raw_traces: ndarray
        Array of traces without background removed
    adj_traces: ndarray
        Array of traces with background removed
    sat_thresh: float
        Saturation threshold. Signals above this are flagged as saturated
    params: GetParameters     
        GET parameters
    Returns
    -------
    ndarray
        Array of traces with background removed and saturated points that are refit.
    """
    tb_grid = np.arange(512)

    # Find saturated points in each trace
    mask = [(trace_i.trace > sat_thresh) for trace_i in raw_traces]

    # Loop over all rows in mask
    for i in range(len(mask)):
        # If the row is all false (i.e no saturation) then continue
        if ~mask[i].any():
            continue

        # Else the row has a saturated point
        else:
            # Calculate an initial guess for the fit parameters
            guess = [max(adj_traces[i].trace), 2, 0, np.mean(tb_grid[mask[i]])]

            try:
                # Fit the portion of the trace without the saturated points
                popt, pcov = curve_fit(
                    SkewedGaussian,
                    tb_grid[~mask[i]],
                    adj_traces[i].trace[~mask[i]],
                    p0=guess,
                    bounds=([0, 0.01, -np.inf, 0], [np.inf, 512, np.inf, 512]),
                )
            except RuntimeError:
                return adj_traces

            # Replace saturated points with fit points
            adj_traces[i].trace[mask[i]] = SkewedGaussian(tb_grid, *popt)[mask[i]]
            adj_traces[i].find_peaks(params)

    return adj_traces


@njit
def preprocess_traces(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    """JIT-ed Method for pre-cleaning the trace data in bulk before doing trace analysis

    These methods are more suited to operating on the entire dataset rather than on a trace by trace basis
    It includes

    - Removal of edge effects in traces (first and last time buckets can be noisy)
    - Baseline removal via fourier transform method (see J. Bradt thesis, pytpc library)

    Parameters
    ----------
    traces: ndarray
        A (n, 512) matrix where n is the number of traces and each row corresponds to a trace. This should be a copied
        array, not a reference to an array in an hdf file
    baseline_window_scale: float
        The scale of the baseline filter used to perform a moving average over the basline

    Returns
    -------
    ndarray
        A new (n, 512) matrix which contains the traces with their baselines removed and edges smoothed
    """
    # Smooth out the edges of the traces
    traces[:, 0] = traces[:, 1]
    traces[:, -1] = traces[:, -2]

    # Remove peaks from baselines and replace with average
    bases: np.ndarray = traces.copy()
    for row in bases:
        mean = np.mean(row)
        sigma = np.std(row)
        mask = row - mean > sigma * 1.5
        row[mask] = np.mean(row[~mask])

    # Create the filter
    window = np.arange(-256.0, 256.0, 1.0)
    fil = np.fft.ifftshift(np.sinc(window / baseline_window_scale))
    transformed = np.fft.fft2(bases, axes=(1,))
    result = np.real(
        np.fft.ifft2(transformed * fil, axes=(1,))
    )  # Apply the filter -> multiply in Fourier = convolve in normal

    return traces - result
