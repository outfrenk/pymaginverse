import numpy as np


def dampingtype(maxdegree: int,
                damp_type: str,
                damp_dipole: bool = False,
                damp_depth: float = 3495 / 6371.2
                ) -> np.ndarray:
    """ Creates spatial or temporal damping array according to type

    Parameters
    ----------
    maxdegree
        maximum order for spherical harmonics model
    damp_type
        style of damping according. Options are:
        Uniform         -> Set all damping to one
        Dissipation     -> Minimize dissipation by minimizing
                           the 2nd norm of Br at the cmb
        Powerseries     -> Minimization condition on power of series used
                           by Löwes
        Gubbins         -> Spatial damping of heat flow at
                           the core mantle boundary (Gubbins et al., 1975)
        Horderiv2cmb    -> Minimization of the integral of the horizontal
                           derivative of B squared
        Energydensity   -> External energy density
        Br2cmb          -> Minimize integral of Br squared over surface
                           at core mantle boundary
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

    """
    func_mapping = {'Uniform': uniform, 'Dissipation': dissipation,
                    'Powerseries': powerseries, 'Gubbins': gubbins,
                    'Horderiv2cmb': horderiv2cmb, 'Br2cmb': br2cmb,
                    'Energydensity': energydensity}
    if damp_type not in func_mapping:
        raise Exception(f'Damping type {damp_type} not found. Exiting...')

    damp_array = np.zeros((maxdegree+1)**2 - 1)
    # allocate the appropriate damping function
    damp_func = func_mapping[damp_type]
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


def dissipation(d: int, rbycmb: float) -> float:
    # according to gubbins and bloxham 1985, bloxham 1987:
    # res = 4*np.pi * (d+1)**2 * (2-d*(d+1)) / ((2*d + 1) * rbycmb**(2*d+4))
    # to be combined with one spline integral (ddt=1)
    res = 4*np.pi * (d+1)**2 * d**4 / ((2*d + 1) * rbycmb**(2*d))
    return res


def powerseries(d: int, rbycmb: float) -> float:
    res = rbycmb**(2*d + 4) * (d+1)
    return res


def gubbins(d: int, rbycmb: float) -> float:
    res = rbycmb**(2*d + 3) * 4*np.pi * (d+1) * (2*d + 1) * (2*d + 3) / d
    return res


def horderiv2cmb(d: int, rbycmb: float) -> float:
    res = rbycmb**(2*d + 6) * d * (d+1)**3 / (2*d + 1)
    return res


def energydensity(d: int, rbycmb: float) -> float:
    res = rbycmb**(2*d + 1) * (d+1) / (2*d + 1)
    return res


def br2cmb(d: int, rbycmb: float) -> float:
    # XXX 4 pi is missing!
    res = rbycmb**(2*d + 4) * (d+1)**2 / (2*d + 1)
    return res
