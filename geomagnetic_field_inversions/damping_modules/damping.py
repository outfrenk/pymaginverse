import numpy as np
from scipy.integrate import newton_cotes
from typing import Literal, Tuple

from .damp_types import dampingtype
from ..tools import bsplines

# list containing the name of damping and its required time derivative
_DList = [['Uniform', 0], ['Dissipation', 0], ['Powerseries', 0],
          ['Gubbins', 0], ['Horderiv2cmb', 0], ['Energydensity', 0],
          ['Br2cmb', 1], ['Br2cmb', 2]]


def damp_matrix(max_degree: int,
                nr_splines: int,
                t_step: float,
                damp_factor: float,
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
    damp_factor
        damping factor to be applied to the total damping matrix (lambda)
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
    spl_degree = 3
    nm_total = (max_degree + 1) ** 2 - 1
    damp_diag = np.zeros(nm_total)

    matrix_diag = np.zeros((2 * spl_degree + 1, nr_splines * nm_total))
    if damp_factor != 0:
        damp_diag = dampingtype(max_degree, _DList[damp_type][0], damp_dipole)
        # start combining interacting splines
        for spl1 in range(nr_splines):  # loop through splines with j
            # loop with spl2 between spl1-spl_degree and spl1+spl_degree
            for spl2 in range(max(spl1 - spl_degree, 0),
                              min(spl1 + spl_degree + 1, nr_splines)):
                # integrate cubic B-Splines
                spl_integral = integrator(spl1, spl2, nr_splines, t_step,
                                          _DList[damp_type][1])
                # place damping in matrix
                matrix_diag[spl2-spl1+spl_degree,
                            spl1*nm_total:(spl1+1)*nm_total
                            ] = damp_factor * spl_integral * damp_diag
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
    spl_degree = 3

    # order of spline after taking derivative
    temp_degree = spl_degree - ddt
    # order of Newton-Cotes integration, depends on spline degree and ddt
    nc_order = 2 * temp_degree
    newcot, error = newton_cotes(nc_order)  # get the weigh factor
    # integration boundaries
    low = int(max(spl1, spl2, spl_degree))
    high = int(min(spl1 + spl_degree, spl2 + spl_degree, nr_splines - 1))
    # necessary to get sum = 1 for weigh factors
    dt = t_step / nc_order

    # create all BSpline derivatives for integration
    spl = bsplines.derivatives(t_step, nc_order + 1, ddt)

    # integrate created splines over time using newton-cotes algorithm
    int_prod = 0
    for t in range(low, high + 1):
        iint_prod = 0
        # go stepwise through the complete integration of one timestep
        for stp in range(nc_order + 1):
            iint_prod += newcot[stp] * spl[t - spl1, stp] * spl[t - spl2, stp]
        int_prod += iint_prod * dt

    return int_prod


def damp_norm(damp_fac: np.ndarray,
              coeff: np.ndarray,
              ddt: int,
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
    ddt
        order of derivation used for damping, 0 - 2
    t_step
        dt of timevector

    Returns
    -------
    norm
        contains the spatial or temporal damping norm per TIME INTERVAL
    """
    spl_degree = 3

    spl = bsplines.derivatives(t_step, 1, ddt).flatten()
    norm = np.zeros(len(coeff))
    # append zero to coeff
    coeffsp = np.vstack((np.zeros((spl_degree, len(coeff[0]))), coeff))
    for t in range(len(coeff)):  # loop through time
        # calculate Gauss coefficient according to derivative spline
        g_spl = np.matmul(spl, coeffsp[t:t+(spl_degree+1)])
        norm[t] = np.dot(damp_fac, g_spl**2)

    return norm[spl_degree-1:] / t_step
