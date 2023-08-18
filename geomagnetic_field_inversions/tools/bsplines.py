import numpy as np
from scipy.interpolate import BSpline


def derivatives(t_step: float,
                steps: int,
                derivative: int
                ) -> np.ndarray:
    """ Takes up to 2nd derivative of a cubic B-Spline

    Parameters
    ----------
    t_step
        times step between the knot points of the spline. Should be constant
    steps
        number of splines during each t_step. depends on order of integration
    derivative
        requested derivative of the cubic B-Spline. Should be between 0 and 2

    Returns
    -------
    spl
        0th, 1st, or 2nd derivative of the cubic B-Spline
    """
    spl_degree = 3
    temp_degree = spl_degree - derivative
    bsp = BSpline.basis_element(np.arange(temp_degree + 2), extrapolate=False)
    bspline = np.zeros((temp_degree + 1, steps))
    for i in range(temp_degree + 1):
        bspline[i] = bsp(np.linspace(i, i+1, steps))
    spl = np.zeros((4, steps))
    if derivative == 0:
        spl[0] = bspline[0]
        spl[1] = bspline[1]
        spl[2] = bspline[2]
        spl[3] = bspline[3]
    elif derivative == 1:
        coef = [1 / t_step, -1 / t_step]
        spl[0] = coef[0] * bspline[0]
        spl[1] = coef[0] * bspline[1] + coef[1] * bspline[0]
        spl[2] = coef[0] * bspline[2] + coef[1] * bspline[1]
        spl[3] = coef[1] * bspline[2]
    elif derivative == 2:
        coef = [1 / t_step**2, -2 / t_step**2, 1 / t_step**2]
        spl[0] = coef[0] * bspline[0]
        spl[1] = coef[0] * bspline[1] + coef[1] * bspline[0]
        spl[2] = coef[1] * bspline[1] + coef[2] * bspline[0]
        spl[3] = coef[2] * bspline[1]
    else:
        raise Exception('This function only implements up to 2nd derivatives')
    return spl
