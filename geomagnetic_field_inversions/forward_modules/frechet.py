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
    frechxyz
        size= stations X 3 X nm_total matrix: contains the frechet coefficients
            first dx, then dy, then dz
    """
    schmidt_total = int((maxdegree+1) * (maxdegree+2) / 2)
    frechxyz = np.zeros(((maxdegree+1)**2 - 1, 3, len(loc)))
    schmidt_p = np.zeros((len(loc), schmidt_total))
    schmidt_dp = np.zeros((len(loc), schmidt_total))
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
        frechxyz[counter, 0] = mult_factor * schmidt_dp[:, index]
        # frechxyz[counter, 1] = 0
        frechxyz[counter, 2] = -mult_factor * (n+1) * schmidt_p[:, index]
        counter += 1
        for m in range(1, n+1):
            # Then the g-elements
            frechxyz[counter, 0] = mult_factor\
                * np.cos(m * loc[:, 1]) * schmidt_dp[:, index+m]
            frechxyz[counter, 1] = m / np.sin(loc[:, 0]) * mult_factor\
                * np.sin(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechxyz[counter, 2] = -mult_factor * (n+1)\
                * np.cos(m * loc[:, 1]) * schmidt_p[:, index+m]
            counter += 1
            # Now the h-elements
            frechxyz[counter, 0] = mult_factor\
                * np.sin(m * loc[:, 1]) * schmidt_dp[:, index+m]
            frechxyz[counter, 1] = -m / np.sin(loc[:, 0]) * mult_factor\
                * np.cos(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechxyz[counter, 2] = -mult_factor * (n+1)\
                * np.sin(m * loc[:, 1]) * schmidt_p[:, index+m]
            counter += 1
    # transpose to get frechet matrix
    return np.swapaxes(frechxyz, 0, 2)


def frechet_types(frechxyz: np.ndarray,
                  forwobs_matrix: np.ndarray,
                  ) -> np.ndarray:
    """
    Calculates the frechet matrix for a specific datatype

    Parameters
    ----------
    frechxyz
        frechet matrix for dx, dy, and dz components produced by frechet_basis
    forwobs_matrix
        Contains the modeled observations.

    Returns
    -------
    frech_matrix
        7*len(stations) X nm_total matrix containing frechet
        coefficients for the specific datatypes. Should probably be used in
        an iterative inversion scheme
    """
    locs = len(frechxyz)
    nm_total = len(frechxyz[0, 0])
    # expand arrays to cover all locations
    # per row first all gh then locs
    width = nm_total*locs  # row width
    txyz = np.repeat(forwobs_matrix[:3], nm_total).reshape(3, width)
    thor = np.repeat(forwobs_matrix[3], nm_total)
    tb_int = np.repeat(forwobs_matrix[4], nm_total)
    dxyz = np.swapaxes(frechxyz, 0, 1).reshape(3, width)
    dhor = (txyz[0]*dxyz[0] + txyz[1]*dxyz[1]) / thor
    # creates frechet for all coefficients, timesteps, and locations
    # rows: all 7 components, per row first all gh then locs
    frech_matrix = np.zeros((7, width))
    frech_matrix[:3] = dxyz
    frech_matrix[3] = dhor
    frech_matrix[4] = (thor*dhor + txyz[2]*dxyz[2]) / tb_int
    frech_matrix[5] = (thor*dxyz[2] - txyz[2]*dhor) / tb_int**2
    frech_matrix[6] = (txyz[0]*dxyz[1] - txyz[1]*dxyz[0]) / thor**2
    # first reshape frech_mat to correspond to datatypes
    # meaning: one row is one station containing every datatype and then
    # Gauss coefficient
    frech_matrix = frech_matrix.reshape(7, locs, nm_total).swapaxes(0, 1)

    return frech_matrix
