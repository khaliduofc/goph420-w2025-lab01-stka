import numpy as np

def integrate_newton(x, f, alg):
    """
    Performs numerical integration using Newton-Cotes rules.

    Parameters:
    x (numpy.ndarray): Array of x-values.
    f (numpy.ndarray): Array of function values corresponding to x.
    alg (str): Algorithm to use ("trap" for trapezoidal, "simp" for Simpson's rule).

    Returns:
    float: Approximate integral value.

    Notes:
    We assume constant step size in x.
    """
    if len(x) != len(f):
        raise ValueError("x and f must have the same length.")

    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        raise ValueError("x values must be equally spaced.")

    if alg == "trap":
        return np.trapezoid(f, x)
    elif alg == "simp":
        if len(x) % 2 == 0:
            raise ValueError("Simpson's rule requires an odd number of points.")
        return (dx[0] / 3) * (f[0] + 4 * np.sum(f[1:-1:2]) + 2 * np.sum(f[2:-2:2]) + f[-1])
    else:
        raise ValueError("Invalid algorithm choice. Use 'trap' or 'simp'.")

def integrate_gauss(f, lims, npts=3):
    """
    Performs numerical integration using Gauss-Legendre quadrature.

    Parameters:
    f (callable): Function to integrate.
    lims (tuple): Lower and upper integration limits.
    npts (int, optional): Number of quadrature points (1 to 5, default=3).

    Returns:
    float: Approximate integral value.
    """
    if not callable(f):
        raise TypeError("f must be a callable function.")

    if len(lims) != 2:
        raise ValueError("lims must contain exactly two values (lower and upper bounds).")

    a, b = lims
    try:
        a, b = float(a), float(b)
    except ValueError:
        raise ValueError("lims must be convertible to float.")

    if npts not in [1, 2, 3, 4, 5]:
        raise ValueError("npts must be in [1, 2, 3, 4, 5].")

    x, w = np.polynomial.legendre.leggauss(npts)
    return (b - a) / 2 * np.sum(w * f((b - a) / 2 * x + (b + a) / 2))
