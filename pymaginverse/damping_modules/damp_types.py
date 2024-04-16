import numpy as np


def dampingtype(maxdegree: int,
                damp_type: str,
                damp_dipole: bool = False,
                damp_depth: float = 3485 / 6371.2
                ) -> np.ndarray:
    """ Creates spatial or temporal damping array according to type

    Parameters
    ----------
    maxdegree
        maximum order for spherical harmonics model
    damp_type
        style of damping according. Options are:
        s_uniform         -> Set all damping to one
        s_energy_diss     -> Minimize dissipation by minimizing
                             the 2nd norm of Br at the cmb
        s_powerseries     -> Minimization condition on power of series used
                             by Löwes
        s_ohmic_heating   -> Spatial damping of heat flow at
                             the core mantle boundary (Gubbins et al., 1975)
        s_smooth_core     -> Minimization of the integral of the horizontal
                             derivative of B squared
        s_min_ext_energy  -> Minimize external energy density
        t_min_vel         -> Minimize integral of Br velocity or acceleration
        t_min_acc            squared over surface at core mantle boundary

    damp_dipole
        If False, damping is not applied to dipole coefficients (first 3).
        If True, dipole coefficients are also damped.
    damp_depth
        scaled depth at which to apply damping. Defaults to CMB/R_earth

    Returns
    -------
    damp_array
        contains damping value for gaussian coefficients. Essentially
        diagonal values of the damping matrix

    Examples
    --------
    >>> dampingtype(2, 's_uniform', True)
    array([1., 1., 1., 1., 1., 1., 1., 1.])
    >>> dampingtype(2, 't_min_acc', False, 1)
    array([0. , 0. , 0. , 1.8, 1.8, 1.8, 1.8, 1.8])
    """
    func_dict = {'s_uniform': uniform, 's_ohmic_heating': ohmic_heating,
                 's_powerseries': powerseries, 's_energy_diss': energy_diss,
                 's_smooth_core': smooth_core, 't_min_vel': min_vel_acc,
                 't_min_acc': min_vel_acc, 's_min_ext_energy': min_ext_energy}
    if damp_type not in func_dict:
        raise Exception(f'Damping type {damp_type} not found. Exiting...')

    damp_array = np.zeros((maxdegree+1)**2 - 1)
    # allocate the appropriate damping function
    damp_func = func_dict[damp_type]
    # radius earth divided by radius cmb
    rbycmb = 1. / damp_depth
    # fill the damping array per degree according to damping type
    # and fill the complete array per g/h component
    counter = 0
    for degree in range(1, maxdegree+1):
        damp_degree = damp_func(degree, rbycmb)
        for count in range(2*degree + 1):  # should start at zero
            if damp_dipole is False and degree == 1:
                damp_array[counter] = 0
            else:
                damp_array[counter] = damp_degree
            counter += 1

    return damp_array


def uniform(d: int, rbycmb: float) -> float:
    return 1.


def energy_diss(d: int, rbycmb: float) -> float:
    res = 4 * np.pi * rbycmb**(2*d+4) * (d+1)**2 * d**4 / (2*d + 1)
    return res


def powerseries(d: int, rbycmb: float) -> float:
    res = 4 * np.pi * rbycmb**(2*d + 4) * (d+1)
    return res


def ohmic_heating(d: int, rbycmb: float) -> float:
    res = 4 * np.pi * rbycmb**(2*d + 3) * (d+1) * (2*d + 1) * (2*d + 3) / d
    return res


def smooth_core(d: int, rbycmb: float) -> float:
    res = 4 * np.pi * rbycmb**(2*d + 6) * d * (d+1)**3 / (2*d + 1)
    return res


def min_ext_energy(d: int, rbycmb: float) -> float:
    res = 4 * np.pi * rbycmb**(2*d + 1) * (d+1) / (2*d + 1)
    return res


def min_vel_acc(d: int, rbycmb: float) -> float:
    # XXX 4 pi is missing!
    res = 4 * np.pi * rbycmb**(2*d + 4) * (d+1)**2 / (2*d + 1)
    return res
