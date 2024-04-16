import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.interpolate import BSpline
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

SPL_DEGREE = 3


def latrad_in_geoc(lat: float,
                   h: float = 0.
                   ) -> Tuple[float, float, float, float]:
    """ Transforms the geodetic latitude to spherical coordinates
    Additionally, calculates the new radius and conversion factors
    source: Stevens et al. Aircraft control and simulation (2015)

    Parameters
    ----------
    lat
        geodetic latitude
    h
        height above geoid (m)

    Returns
    -------
    new_lat
        converted latitude
    new_rad
        recalculated radius
    cd
        conversion factor for calculating geodetic magnetic x,z components.
        If geocentric, cd is 1
    sd
        conversion factor for calculating geodetic magnetic x,z components.
        If geocentric, sd is 0
    """
    # major axis earth (m)
    maj_axis = 6378137.0
    # minor axis earth (m)
    min_axis = 6356752
    # flattening
    flatt = (maj_axis-min_axis) / min_axis
    # eccentricity squared
    ecc2 = flatt * (2-flatt)
    # prime vertical radius of curvature
    Rn = maj_axis / np.sqrt(1 - ecc2 * np.sin(lat)**2)
    # helpful parametrization
    n = ecc2 * Rn / (Rn+h)

    # calculate new geocentric latitude
    new_lat = np.arctan((1-n) * np.tan(lat))
    # calculate new geocentric radius in km!
    new_rad = (Rn+h) * np.sqrt(1 - n * (2-n) * np.sin(lat)**2) * 1e-3
    # calculate changes in vector
    cd = np.cos(lat-new_lat)
    sd = np.sin(lat-new_lat)
    return new_lat, new_rad, cd, sd


def frechet_in_geoc(dx: np.ndarray,
                    dz: np.ndarray,
                    cd: np.ndarray,
                    sd: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Transforms geodetic frechet components to geocentric

    Parameters
    ----------
    dx
        polar component of the frechet matrix
    dz
        radial component of the frechet matrix
    cd
        cosine of latitude difference
    sd
        sine of latitude difference
    """
    dxnew = dx * cd[:, np.newaxis] + dz * sd[:, np.newaxis]
    dznew = dz * cd[:, np.newaxis] - dx * sd[:, np.newaxis]
    return dxnew, dznew


def calc_stdev(path: Path,
               degree: int,
               save_covar: bool = False,
               save_res: bool = False,
               verbose: bool = True):
    """ Function to calculate and save standard deviations

    Parameters
    ----------
    path
        path to location of normal_equation matrix + where to save covariance
    degree
        degree of the spherical model.
    save_covar
        boolean indicating whether to save covariance matrix. default is False
    save_res
        boolean indicating whether to save resolution matrix. default is False
    verbose
        verbosity flag
    """
    # TODO: Optimize if possible
    normal_eq = np.load(path / 'forward_matrix.npy')
    damp = np.load(path / 'damp_matrix.npy')
    norm_damp = normal_eq.copy()
    norm_damp[norm_damp.shape[0] - damp.shape[0]:] += damp
    if save_res:
        if verbose:
            print('Calculating resolution matrix')
        chol = cholesky_banded(norm_damp)
        n, d = normal_eq.shape
        res_mat = np.zeros((d, d))
        n_range = np.arange(n)
        rows = np.zeros(2 * len(n_range) - 1)
        rows[:n] = n_range
        rows[n:] = n_range[-2::-1]
        for i in tqdm(range(d)):
            rhs = np.zeros(d)
            row_sub = rows[max(n-1-i, 0):min(n-1+d-i, 2*n - 1)]
            col_sub = np.zeros_like(row_sub)
            col_sub[min(i, n-1):] = np.arange(i, min(i+n, d))
            col_sub[:min(i, n-1)] = i
            idx = (col_sub + row_sub * d).astype(int)
            rhs[max(i-n+1, 0):min(i+n, d)] = normal_eq.flatten()[idx]
            res_mat[i] = cho_solve_banded((chol, False), rhs)
        np.save(path / 'resolution_mat', res_mat)
        del res_mat

    del normal_eq, damp
    if verbose:
        print('Start inversion for covariance')
    chol = cholesky_banded(norm_damp)
    n, d = norm_damp.shape
    covar = np.zeros((d, d))
    for i in tqdm(range(d)):
        rhs = np.zeros(d)
        rhs[i] = 1
        covar[i] = cho_solve_banded((chol, False), rhs)
    if verbose:
        print('Finished inversion, calculating std')
    nm_total = (degree+1)**2 - 1
    std = np.sqrt(np.diag(covar)).reshape(-1, nm_total)
    if verbose:
        print('Saving')
    np.save(path / 'std', std)
    if save_covar:
        np.save(path / 'covar', covar)
    if verbose:
        print('Saving finished')


def calc_spectra(coeff: np.ndarray,
                 maxdegree: int,
                 t_step: float,
                 cmb: bool = False):
    """ Calculates the power spectrum and secular variation spectrum (Löwes)

    Parameters
    ----------
    coeff
        Gauss coefficients, each row containing the Gauss coefficients per
        spline
    maxdegree
         Spherical degree of modeled field
    t_step
        timestep size
    cmb
        Whether to calculate the powerspectra at the cmb (True)
         or surface (False; default)

    Returns
    -------
    sum_coeff_pow, sum_coeff_sv
        powerspectrum and sv spectrum
    """
    steps_between = len(coeff) - SPL_DEGREE
    if cmb:
        depth = 6371.2 / 3485.0
    else:
        depth = 1

    spl1 = BSpline.basis_element(np.arange(SPL_DEGREE+2) * t_step,
                                 extrapolate=False).derivative(1)(
           np.arange(0, SPL_DEGREE+1) * t_step)
    spl0 = BSpline.basis_element(np.arange(SPL_DEGREE+2) * t_step,
                                 extrapolate=False).derivative(0)(
           np.arange(0, SPL_DEGREE+1) * t_step)
    coeff_big = np.vstack((np.zeros((SPL_DEGREE, len(coeff[0]))), coeff))
    coeff_sv = np.zeros_like(coeff)
    coeff_pow = np.zeros_like(coeff)
    for t in range(len(coeff)):
        # calculate Gauss coefficient according to derivative spline
        coeff_pow[t] = np.matmul(spl0, coeff_big[t:t + SPL_DEGREE+1])**2
        coeff_sv[t] = np.matmul(spl1, coeff_big[t:t + SPL_DEGREE+1])**2
    coeff_pow = np.sum(coeff_pow[SPL_DEGREE-1:], axis=0) / steps_between
    coeff_sv = np.sum(coeff_sv[SPL_DEGREE-1:], axis=0) / steps_between
    counter = 0
    sum_coeff_pow = np.zeros(maxdegree)
    sum_coeff_sv = np.zeros(maxdegree)
    for l in range(maxdegree):
        for m in range(2*l + 1):
            sum_coeff_pow[l] += coeff_pow[counter] * (l+2) * depth**(2*(l+1)+4)
            sum_coeff_sv[l] += coeff_sv[counter] * (l+2) * depth**(2*(l+1)+4)
            counter += 1

    return sum_coeff_pow, sum_coeff_sv
