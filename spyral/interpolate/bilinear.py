import numpy as np
import math
from numba import njit, int32, float64, boolean
from numba.experimental import jitclass


@njit
def clamp(value: float | int, low: float | int, hi: float | int) -> float | int:
    """Clamp a value to a range

    Parameters
    ----------
    value: float | int
        Value to be clamped
    low: float | int
        Bottom of the clamp range
    hi: float | int
        Top of the clamp range

    Returns
    -------
    float | int
        Clamped value
    """
    return max(low, min(value, hi))


# To use numba with a class we need to declare the types of all members of the class
# and use the @jitclass decorator
bilinear_spec = [
    ("x_min", float64),
    ("x_max", float64),
    ("x_bins", int32),
    ("x_width", float64),
    ("y_min", float64),
    ("y_max", float64),
    ("y_bins", int32),
    ("y_width", float64),
    ("values", float64[:, :, :]),
    ("nan", boolean),
]


@jitclass(spec=bilinear_spec)
class BilinearInterpolator:
    """A JIT-ed bilinear interpolation class

    Interpolator for regularly spaced grid, where axes are given in strictly ascending order.
    Values outside the interpolation range can either be clamped to the interpolation range or result in a
    NaN value. Interpolates functions like f(x, y) -> z where z can be a vector or scalar.

    We use numba to just-in-time compile these methods, which results in dramatic speed increases, on the order of a factor of 50

    The grid is NxMxP shaped. N is the length of the grid in x, M the length in y, and P the length of the function output.

    Attributes
    ----------
    x_min: float
        Minimum value of the x-coordinate for the grid
    x_max: float
        Maximum value fo the x-coordinate for the grid
    x_bins: int
        Number of bins (cells) in the x-coordinate for the grid
    y_min: float
    y   Minimum value of the y-coordinate for the grid
    y_max: float
    y   Maximum value fo the y-coordinate for the grid
    y_bins: int
        Number of bins (cells) in the y-coordinate for the grid
    values: ndarray
        The NxMxP grid of data to interpolate on
    nan: bool
        The policy of handling requests to extrapolate on the grid. If nan is True,
        requests to extrapolate will return arrays of NaN values. Otherwise the requests
        are clamped to the edge of the grid

    Methods
    -------
    BilinearInterpolator(x_min: float, x_max: float, x_bins: int, y_min: float, y_max: float, y_bins: int, data: ndarray, nan: bool=True)
        Construct the interpolator.
    get_edges_x(value: float) -> tuple[int, float, int, float]
        Get the edges of the grid cell in the x-coordinate in coordinate and bin space for a given value
    get_edges_y(value: float) -> tuple[int, float, int, float]
        Get the edges of the grid cell in the y-coordinate in coordinate and bin space for a given value
    check_values_shape()
        Internal consistency check of the grid. Raises an Exception if check fails.
    interpolate(x: float, y: float) -> np.ndarray
        Interpolate on a given coordinate (x,y)
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        x_bins: int,
        y_min: float,
        y_max: float,
        y_bins: int,
        data: np.ndarray,
        nan: bool = True,
    ):
        """Construct the BilinearInterpolator

        Parameters
        ----------
        x_min: float
            Minimum value of the x-coordinate for the grid
        x_max: float
            Maximum value fo the x-coordinate for the grid
        x_bins: int
            Number of bins (cells) in the x-coordinate for the grid
        y_min: float
            Minimum value of the y-coordinate for the grid
        y_max: float
            Maximum value fo the y-coordinate for the grid
        y_bins: int
            Number of bins (cells) in the y-coordinate for the grid
        data: ndarray
            The NxMxP grid of data to interpolate on
        nan: bool
            The policy of handling requests to extrapolate on the grid. If nan is True,
            requests to extrapolate will return arrays of NaN values. Otherwise the requests
            are clamped to the edge of the grid. Default is True.

        Returns
        -------
        BilinearInterpolator
            An instance of the class
        """
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.x_bins: int = x_bins
        self.x_width: float = (self.x_max - self.x_min) / float(self.x_bins)
        self.y_min: float = y_min
        self.y_max: float = y_max
        self.y_bins: int = y_bins
        self.y_width: float = (self.y_max - self.y_min) / float(self.y_bins)
        self.values: np.ndarray = data
        self.nan: bool = nan
        self.check_values_shape()

    def check_values_shape(self):
        """Internal consistency check of the grid. Raises an Exception if check fails.

        Check to make sure all of the data given to the interpolator makes sense.

        """
        values_shape = self.values.shape
        if len(values_shape) < 3:
            print(
                f"The values given to BilinearInterpolator do not have the correct dimensionality! Given {values_shape}, requires a minimum 3 dimensions"
            )
            raise Exception
        if values_shape[0] != self.x_bins:
            print(
                f"The shape of the values given to BilinearInterpolator along the x-axis does not match the given x-axis! axis={self.x_bins} values={values_shape[0]}"
            )
            raise Exception
        if values_shape[1] != self.y_bins:
            print(
                f"The shape of the values given to BilinearInterpolator along the y-axis does not match the given y-axis! axis={self.y_bins} values={values_shape[1]}"
            )
            raise Exception

    def get_edges_x(self, value: float) -> tuple[int, float, int, float]:
        """Get the edges of the grid cell in the x-coordinate in coordinate and bin space for a given value

        Parameters
        ----------
        value: float
            The value in x-coorindates for which edges should be found

        Returns
        -------
        tuple[int, float, int, float]
            A four-member tuple of (low x-bin number, low x-coorindate, high x-bin number, high x-coordinate)
            which describes the edges of the grid cell
        """
        bin_low = math.floor((value - self.x_min) / float(self.x_width))
        edge_low = self.x_min + bin_low * self.x_width
        bin_hi = min(bin_low + 1, self.x_bins - 1)
        edge_hi = self.x_min + bin_hi * self.x_width
        return (bin_low, edge_low, bin_hi, edge_hi)

    def get_edges_y(self, value: float) -> tuple[int, float, int, float]:
        """Get the edges of the grid cell in the y-coordinate in coordinate and bin space for a given value

        Parameters
        ----------
        value: float
            The value in y-coorindates for which edges should be found

        Returns
        -------
        tuple[int, float, int, float]
            A four-member tuple of (low y-bin number, low y-coorindate, high y-bin number, high y-coordinate)
            which describes the edges of the grid cell
        """
        bin_low = math.floor((value - self.y_min) / float(self.y_width))
        edge_low = self.y_min + bin_low * self.y_width
        bin_hi = min(bin_low + 1, self.y_bins - 1)
        edge_hi = self.y_min + bin_hi * self.y_width
        return (bin_low, edge_low, bin_hi, edge_hi)

    def interpolate(self, x: float, y: float) -> np.ndarray:
        """Interpolate on a given coordinate (x,y)

        Parameters
        ----------
        x: float
            The x-coordinate of the point to interpolate
        y: float
            The y-coordinate of the point to interpolate

        Returns
        -------
        ndarray
            The interpolated value. If the extrapolation policy was set to NaN,
            this can contain NaN values. Otherwise the requests
            are clamped to the edge of the grid
        """
        if self.nan and (
            x > self.x_max or x < self.x_min or y < self.y_min or y > self.y_max
        ):
            return np.full(self.values.shape[2], np.nan)

        x = clamp(x, self.x_min, self.x_max)
        y = clamp(y, self.y_min, self.y_max)
        x1_bin, x1, x2_bin, x2 = self.get_edges_x(x)
        y1_bin, y1, y2_bin, y2 = self.get_edges_y(y)

        q11 = self.values[x1_bin, y1_bin]
        q12 = self.values[x1_bin, y2_bin]
        q21 = self.values[x2_bin, y1_bin]
        q22 = self.values[x2_bin, y2_bin]
        x2x = x2 - x
        y2y = y2 - y
        xx1 = x - x1
        yy1 = y - y1

        if x2 == x1 and y1 == y2:  # On a corner
            return q11
        elif x2 == x1:  # At xlim
            return (q11 * yy1 + q12 * y2y) / (y2 - y1)
        elif y1 == y2:  # At ylim
            return (q11 * xx1 + q21 * x2x) / (x1 - x2)
        else:  # In a cell
            return (
                q11 * (x2x) * (y2y)
                + q21 * (xx1) * (y2y)
                + q12 * (x2x) * (yy1)
                + q22 * (xx1) * (yy1)
            ) / ((x2 - x1) * (y2 - y1))
