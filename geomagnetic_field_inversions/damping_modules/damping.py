import numpy as np
from scipy.integrate import newton_cotes
from scipy.interpolate import BSpline
from typing import Literal
from .damp_types import dampingtype
_DampingMethods = Literal['Uniform', 'Dissipation', 'Powerseries', 'Gubbins',
                          'Horderiv2cmb', 'Br2cmb', 'Energydensity']


def damp_matrix(max_degree: int,
                nr_splines: int,
                t_step: float,
                damp_factor: float,
                damp_type: _DampingMethods,
                ddt: int,
                damp_dipole: bool = True,
                ) -> np.ndarray:
    """ Creates spatial and temporal damping matrices

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
        damping type to be applied
    ddt
        derivative of B-Splines to be applied
    damp_dipole
        boolean indicating whether to damp dipole coefficients or not.
        Default is set to False.

    Returns
    -------
    matrix
        damping matrix
    """
    spl_degree = 3
    nm_total = (max_degree+1)**2 - 1
    matrix = np.zeros((nm_total * nr_splines, nm_total * nr_splines))

    if damp_factor != 0:
        damp_diag = dampingtype(max_degree, damp_type, damp_dipole)
        # start combining interacting splines
        for spl1 in range(nr_splines):  # loop through splines with j
            # loop with spl2 between spl1-spl_degree and spl1+spl_degree
            for spl2 in range(max(0, spl1-spl_degree),
                              min(spl1+spl_degree+1, nr_splines)):
                # integrate cubic B-Splines
                spl_integral = integrator(spl1, spl2, nr_splines, t_step, ddt)
                # place damping in matrix
                matrix[spl1 * nm_total:(spl1+1) * nm_total,
                       spl2 * nm_total:(spl2+1) * nm_total] =\
                    damp_factor * spl_integral * np.diag(damp_diag)
    return matrix


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
    # get the correct spline
    bspline = BSpline.basis_element(np.arange(temp_degree+2),
                                    extrapolate=False)
    # order of Newton-Cotes integration, depends on spline degree and ddt
    nc_order = 2 * temp_degree
    newcot, error = newton_cotes(nc_order)  # get the weigh factor
    # integration boundaries
    low = int(max(spl1, spl2, spl_degree))
    high = int(min(spl1+spl_degree, spl2+spl_degree, nr_splines-1))
    # necessary to get sum = 1 for weigh factors
    dt = t_step / nc_order

    # create all BSplines for integration
    bspline_matrix = np.zeros((temp_degree+1, nc_order+1))
    for i in range(temp_degree+1):
        # old, BSplinematrix is of course symmetric!
        # bspline_matrix[i] = bspline(np.linspace(i, i+1, nc_order+1))[::-1]
        bspline_matrix[i] = bspline(np.linspace(i, i+1, nc_order+1))

    # if loop for different derivative of splines (between 0 and 2)
    if ddt == 0:
        spl = bspline_matrix
        # old
        # for t in range(int(low), int(high + 1)):
        # int_prod += np.sum(newcot * bspline_matrix[(spl1 + spl_degree) - t]
        #                    * bspline_matrix[(spl2 + spl_degree) - t]) * dt
    elif ddt == 1:
        # coefficients originating from derivative splines
        # see A practical guide to splines by Boor (2000) for details
        coeff = [1 / t_step, -1 / t_step]
        spl = np.zeros((4, nc_order + 1))
        # old
        # spl[0, :] = coeff[0] * bspline_matrix[2, :]
        spl[0, :] = coeff[0] * bspline_matrix[0, :]
        # old
        # spl[1, :] = coeff[0] * bspline_matrix[1, :] \
        #             + coeff[1] * bspline_matrix[2, :]
        spl[1, :] = coeff[0] * bspline_matrix[1, :]\
                    + coeff[1] * bspline_matrix[0, :]
        # old
        # spl[2, :] = coeff[0] * bspline_matrix[0, :] \
        #             + coeff[1] * bspline_matrix[1, :]
        spl[2, :] = coeff[0] * bspline_matrix[2, :] \
                    + coeff[1] * bspline_matrix[1, :]
        # old
        # spl[3, :] = coeff[1] * bspline_matrix[0, :]
        spl[3, :] = coeff[1] * bspline_matrix[2, :]
    elif ddt == 2:
        coeff = [1 / t_step**2, -2 / t_step**2, 1 / t_step**2]
        spl = np.zeros((4, nc_order + 1))
        # old
        # spl[0, :] = coeff[0] * bspline_matrix[1, :]
        spl[0, :] = coeff[0] * bspline_matrix[0, :]
        # old
        # spl[1, :] = coeff[0] * bspline_matrix[0, :] \
        #             + coeff[1] * bspline_matrix[1, :]
        spl[1, :] = coeff[0] * bspline_matrix[1, :] \
                    + coeff[1] * bspline_matrix[0, :]
        # old
        # spl[2, :] = coeff[1] * bspline_matrix[0, :] \
        #             + coeff[2] * bspline_matrix[1, :]
        spl[2, :] = coeff[1] * bspline_matrix[1, :] \
                    + coeff[2] * bspline_matrix[0, :]
        # old
        # spl[3, :] = coeff[2] * bspline_matrix[0, :]
        spl[3, :] = coeff[2] * bspline_matrix[1, :]
    else:
        raise ValueError(f'ddt option {ddt} does not exist! '
                         'Choose ddt between 0 and 2')
    # integrate created splines over time using newton-cotes algorithm
    int_prod = 0
    for t in range(low, high+1):
        iint_prod = 0
        # go stepwise through the complete integration of one timestep
        for stp in range(nc_order+1):
            iint_prod += newcot[stp] * spl[t-spl1, stp] * spl[t-spl2, stp]
        int_prod += iint_prod * dt

    return int_prod
