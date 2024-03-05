import numpy as np
from numba import njit


@njit
def generate_circle_points(
    center_x: float, center_y: float, radius: float
) -> np.ndarray:
    """Jit-ed generate points on a circle

    Given a center and a radius, get the points on a circle

    Parameters
    ----------
    center_x: float
        Circle center x-coordinate
    center_y: float
        Circle center y-coordinate
    radius: float
        Circle radius

    Returns
    -------
    ndarray
        A Nx2 array where the first column is the x-coordinate of a point on the circle
        and the second column is the y-coordinate of a point on the circle

    """
    theta = np.linspace(0.0, 2.0 * np.pi, 100000)
    array = np.zeros(shape=(len(theta), 2))
    array[:, 0] = center_x + np.cos(theta) * radius
    array[:, 1] = center_y + np.sin(theta) * radius
    return array


@njit
def least_squares_circle(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    """Jit-ed least-squares circle fit

    Implementation of analytic least squares circle fit. Taken from the scipy cookbooks.

    Parameters
    ----------
    x: ndarray
        List of all x position coordinates to be fit
    y: ndarray
        list of all y position coordinates to be fit

    Returns
    -------
    tuple[float, float, float, float]
        A four member tuple containing the center x-coordinate, the center y-coordinate, the radius, and the RMSE (in that order)
        These are NaN if the matrix is singular
    """
    mean_x = x.mean()
    mean_y = y.mean()

    # Reduced coordinates
    u = x - mean_x
    v = y - mean_y

    # linear system defining the center (uc, vc) in reduced coordinates
    # Suu * uc + Suv * vc = (Suuu + Suvv)/2
    # Suv * uc + Svv * vc = (Suuv + Svvv)/2
    # => A * c = B
    Suv = np.sum(u * v)
    Suu = np.sum(u**2.0)
    Svv = np.sum(v**2.0)
    Suuv = np.sum(u**2.0 * v)
    Suvv = np.sum(u * v**2.0)
    Suuu = np.sum(u**3.0)
    Svvv = np.sum(v**3.0)

    matrix_a = np.array([[Suu, Suv], [Suv, Svv]])
    matrix_b = np.array([(Suuu + Suvv) * 0.5, (Suuv + Svvv) * 0.5])
    c = None
    try:
        c = np.linalg.solve(matrix_a, matrix_b)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

    xc = c[0] + mean_x
    yc = c[1] + mean_y
    radii = np.sqrt((x - xc) ** 2.0 + (y - yc) ** 2.0)
    mean_radius = np.mean(radii)
    residual = np.sum((radii - mean_radius) ** 2.0)
    return (xc, yc, mean_radius, residual)
