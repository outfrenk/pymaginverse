import numpy as np
import pyshtools as pysh
from .fwtools import forward_obs


def frechet_basis(loc: np.ndarray,
                  maxdegree: int
                  ) -> np.ndarray:
    """
    Calculates the frechet matrix for the given stations and maximum degree
    Parameters
    ----------
    loc
        coordinates of stations. Each row contains:
            colatitude in radians
            longitude in radians
            radius in km
    maxdegree
        Maximum spherical degree

    Returns
    -------
    frechxyz.transpose()
        size= 3 * stations X nm_total matrix: contains the frechet coefficients
            first len(loc) rows contain dx
            len(loc) to 2*len(loc) rows contain dy
            2*len(loc) to 3*len(loc) rows contain dz
    """
    schmidt_total = int((maxdegree+1) * (maxdegree+2) / 2)
    ll = len(loc)
    frechxyz = np.zeros(((maxdegree+1)**2 - 1, 3*len(loc)))
    schmidt_p = np.zeros((ll, schmidt_total))
    schmidt_dp = np.zeros((ll, schmidt_total))
    for i, coord in enumerate(loc):
        schmidt_p[i], schmidt_dp[i] = \
            pysh.legendre.PlmSchmidt_d1(maxdegree, np.cos(coord[0]))
        schmidt_dp[i] *= -np.sin(coord[0])
    counter = 0
    # dx, dy, dz in separate rows to increase speed
    for n in range(1, maxdegree+1):
        index = int(n * (n+1) / 2)
        mult_factor = (6371.2 / loc[:, 2]) ** (n+1)
        # first g_n^0
        frechxyz[counter, :ll] = mult_factor * schmidt_dp[:, index]
        # frechxyz[counter, ll:2*ll] = 0
        frechxyz[counter, 2*ll:] = -mult_factor * (n+1) * schmidt_p[:, index]
        counter += 1
        for m in range(1, n+1):
            # Then the g-elements
            frechxyz[counter, :ll] = mult_factor\
                * np.cos(m * loc[:, 1]) * schmidt_dp[:, index+m]
            frechxyz[counter, ll:2*ll] = m / np.sin(loc[:, 0]) * mult_factor\
                * np.sin(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechxyz[counter, 2*ll:] = -mult_factor * (n+1)\
                * np.cos(m * loc[:, 1]) * schmidt_p[:, index+m]
            counter += 1
            # Now the h-elements
            frechxyz[counter, :ll] = mult_factor\
                * np.sin(m * loc[:, 1]) * schmidt_dp[:, index+m]
            frechxyz[counter, ll:2*ll] = -m / np.sin(loc[:, 0]) * mult_factor\
                * np.cos(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechxyz[counter, 2*ll:] = -mult_factor * (n+1)\
                * np.sin(m * loc[:, 1]) * schmidt_p[:, index+m]
            counter += 1
    # transpose to get frechet matrix
    return frechxyz.transpose()


def frechet_types(frechxyz: np.ndarray,
                  types_sort: np.ndarray,
                  forwobs_matrix: np.ndarray = None,
                  coeff:np.ndarray = None
                  ) -> np.ndarray:
    """
    Calculates the frechet matrix for a specific datatype and all timesteps

    Parameters
    ----------
    frechxyz
        frechet matrix for dx, dy, and dz components produced by frechet_basis
    types_sort
        Tells the type of data by providing an index to every row in either
        forwobs_matrix or data_matrix
    forwobs_matrix
        Contains the modeled observations. If not provided is calculated with
        forward_obs function
    coeff
        Gauss coefficients. Each row contains the coefficients of one timestep.
        Has to be provided if forwobs_matrix is not inputted.

    Returns
    -------
    frech_matrix
        7*len(stations) X nm_total*len(time array) matrix containing frechet
        coefficients for the specific datatypes. Should probably be used in
        an iterative inversion scheme
    """
    if forwobs_matrix is None:
        if coeff is None:
            raise Exception('Since forward observations are unknown, '
                            'please provide Gauss coefficients')
        else:
            forwobs_matrix = forward_obs(coeff, frechxyz, False)
    locs = len(frechxyz) // 3
    nm_total = len(frechxyz[0])
    times = len(forwobs_matrix[0]) // locs
    # expand arrays to cover all time
    # per row first all gh then new time then locs
    width = nm_total*times*locs  # row width
    txyz = np.repeat(forwobs_matrix[:3], nm_total).reshape(3, width)
    thor = np.repeat(forwobs_matrix[3], nm_total)
    tb_int = np.repeat(forwobs_matrix[4], nm_total)
    dxyz = np.tile(frechxyz, times).reshape(3, width)
    dhor = (txyz[0]*dxyz[0] + txyz[1]*dxyz[1]) / thor
    # creates frechet for all coefficients, timesteps, and locations
    # rows: all 7 components, per row first all gh then new time then locs
    frech_matrix = np.zeros((7, width))
    frech_matrix[:3] = dxyz
    frech_matrix[3] = dhor
    frech_matrix[4] = (thor*dhor + txyz[2]*dxyz[2]) / tb_int
    frech_matrix[5] = (thor*dxyz[2] - txyz[2]*dhor) / tb_int**2
    frech_matrix[6] = (txyz[0]*dxyz[1] - txyz[1]*dxyz[0]) / thor**2
    # now select useful rows by data_type
    # first reshape forwobs_matrix and frech_mat to correspond to datatypes
    # meaning: one row is one datatype of station at every time
    frech_matrix = frech_matrix.reshape(
        7, locs, nm_total*times).swapaxes(0, 1).reshape(7*locs, nm_total*times)

    return frech_matrix[types_sort]
