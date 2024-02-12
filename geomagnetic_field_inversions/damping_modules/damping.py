import numpy as np
from scipy.integrate import newton_cotes
from scipy.interpolate import BSpline
from typing import Tuple

from .damp_types import dampingtype

SPL_DEGREE = 3
# list containing the name of damping and its required time derivative
_DList = [['Uniform', 0], ['Dissipation', 0], ['Powerseries', 0],
          ['Gubbins', 0], ['Horderiv2cmb', 0], ['Energydensity', 0],
          ['Br2cmb', 1], ['Br2cmb', 2]]


def damp_matrix(max_degree: int,
                nr_splines: int,
                t_step: float,
                damp_type: int,
                damp_dipole: bool = True,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates spatial and temporal damping matrices through diagonals

    Parameters
    ----------
    max_degree
        maximum order for spherical harmonics model
    nr_splines
        amount of splines, at least length(time array) + degree B-Splines - 1
    t_step
        time step of time array
    damp_type
        integer of damping type to be applied (see _DList)
    damp_dipole
        boolean indicating whether to damp dipole coefficients or not.
        Default is set to False.

    Returns
    -------
    matrix
        damping matrix
    damp_diag
        damping parameters per degree (and order)
    """
    nm_total = (max_degree + 1) ** 2 - 1
    damp_diag = np.zeros(nm_total)

    matrix_diag = np.zeros((2 * SPL_DEGREE + 1, nr_splines * nm_total))
    damp_diag = dampingtype(max_degree, _DList[damp_type][0], damp_dipole)
    # start combining interacting splines
    for spl1 in range(nr_splines):  # loop through splines with j
        # loop with spl2 between spl1-spl_degree and spl1+spl_degree
        for spl2 in range(max(spl1 - SPL_DEGREE, 0),
                          min(spl1 + SPL_DEGREE + 1, nr_splines)):
            # integrate cubic B-Splines
            spl_integral = integrator(spl1, spl2, nr_splines, t_step,
                                      _DList[damp_type][1])
            # place damping in matrix
            matrix_diag[spl2 - spl1 + SPL_DEGREE,
                        spl1 * nm_total:(spl1 + 1) * nm_total
                        ] = spl_integral * damp_diag
    return matrix_diag, damp_diag


def integrator(spl1: int,
               spl2: int,
               nr_splines: int,
               t_step: float,
               ddt: int
               ) -> float:
    """ Integrates inputted splines or derivatives for given time interval
    It automatically integrates over the time that is covered by both splines

    Parameters
    ----------
    spl1
        index of first spline
    spl2
        index of second spline
    nr_splines
        total number of splines
    t_step
        time step of time array
    ddt
        used derivative of B-Spline; should be an integer between 0 and 2

    Returns
    -------
    int_prod
        integration product of inputted splines or spline derivatives
    """
    # order of spline after taking derivative
    temp_degree = SPL_DEGREE - ddt
    # order of Newton-Cotes integration, depends on spline degree and ddt
    nc_order = 2 * temp_degree
    newcot, error = newton_cotes(nc_order)  # get the weigh factor
    # integration boundaries
    low = int(max(spl1, spl2, SPL_DEGREE))
    high = int(min(spl1 + SPL_DEGREE, spl2 + SPL_DEGREE, nr_splines - 1))
    # necessary to get sum = 1 for weigh factors
    dt = t_step / nc_order

    # create all cubic BSpline (derivatives) for integration
    bspline = BSpline.basis_element(np.arange(SPL_DEGREE+2) * t_step,
                                    extrapolate=False).derivative(ddt)

    # integrate created splines over time using newton-cotes algorithm
    int_prod = 0
    for t in range(low, high + 1):
        # go through the complete integration of one timestep
        int_prod += np.sum(newcot * bspline(
            np.linspace(t-spl1, t-spl1+1, nc_order+1) * t_step) * bspline(
            np.linspace(t-spl2, t-spl2+1, nc_order+1) * t_step)) * dt

    return int_prod


def damp_norm(damp_fac: np.ndarray,
              coeff: np.ndarray,
              damp_type: int,
              t_step: float,
              ) -> np.ndarray:
    """
    Calculates the spatial or temporal damping norm

    Parameters
    ----------
    damp_fac
        damping diagonal as produced by damp_types
    coeff
        splined Gauss coefficients of one time per row
    damp_type
        integer of damping type to be applied (see _DList)
    t_step
        dt of timevector

    Returns
    -------
    norm
        contains the spatial or temporal damping norm per TIME INTERVAL
        NOTE: DOES NOT NORMALIZE!
    """
    ddt = _DList[damp_type][1]
    spl = BSpline.basis_element(np.arange(SPL_DEGREE+2) * t_step,
                                extrapolate=False).derivative(ddt)(
          np.arange(0, SPL_DEGREE+1) * t_step)
    norm = np.zeros(len(coeff) - (SPL_DEGREE-1))
    norm[0] = np.dot(damp_fac, np.matmul(spl[1:], coeff[:3])**2)
    for t in range(1, len(coeff) - (SPL_DEGREE-1)):  # loop through time
        # calculate Gauss coefficient according to derivative spline
        g_spl = np.matmul(spl, coeff[t-1:t+SPL_DEGREE])
        norm[t] = np.dot(damp_fac, g_spl**2)

    return norm
