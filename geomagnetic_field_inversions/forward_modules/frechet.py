import numpy as np
import pyshtools as pysh


def frechet_formation(loc: np.ndarray,
                      maxdegree: int,
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
    frechmat.transpose()
        # stations X 3 * nm_total - matrix: contains the frechet coefficients
            first nm_total columns contain dx
            nm_total to 2*nm_total columns contain dy
            2*nm_total to 3*nm_total columns contain dz
    """
    nm_total = (maxdegree+1)**2 - 1
    frechmat = np.zeros((3*nm_total, len(loc)))
    schmidt_p = np.zeros((len(loc), nm_total))
    schmidt_dp = np.zeros((len(loc), nm_total))
    for i, coord in enumerate(loc):
        schmidt_p[i], schmidt_dp[i] = \
            pysh.legendre.PlmSchmidt_d1(maxdegree+1, np.cos(coord[0]))
        schmidt_dp[i] *= -np.sin(coord[0])
    counter = 0
    # dx, dy, dz in separate rows to increase speed
    for n in range(1, maxdegree+1):
        index = int(n * (n+1) / 2)
        mult_factor = (6371.2 / loc[:, 2]) ** (n+1)
        # first g_1^0
        frechmat[counter] = mult_factor * schmidt_dp[:, index]
        # frechmat[counter+nm_total] = 0
        frechmat[counter+2*nm_total] = -mult_factor * (n+1)\
            * schmidt_p[:, index]
        counter += 1
        for m in range(1, n+1):
            # Then the g-elements
            frechmat[counter] = mult_factor * schmidt_dp[:, index+m]\
                * np.cos(m * loc[:, 1])
            frechmat[counter+nm_total] = m / np.sin(loc[:, 0]) * mult_factor\
                * np.sin(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechmat[counter+2*nm_total] = -mult_factor * (n+1)\
                * schmidt_p[:, index + m] * np.cos(m * loc[:, 1])
            counter += 1
            # Now the h-elements
            frechmat[counter] = mult_factor * schmidt_dp[:, index+m]\
                * np.sin(m * loc[:, 1])
            frechmat[counter+nm_total] = -m / np.sin(loc[:, 0])\
                * mult_factor * np.cos(m * loc[:, 1]) * schmidt_p[:, index+m]
            frechmat[counter+2*nm_total] = -mult_factor * (n+1) *\
                schmidt_p[:, index+m] * np.sin(m * loc[:, 1])
            counter += 1
    # transpose to get frechet matrix
    return frechmat.transpose()
