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
    frechxyz.transpose()
        # stations X 3 * nm_total - matrix: contains the frechet coefficients
            first nm_total columns contain dx
            nm_total to 2*nm_total columns contain dy
            2*nm_total to 3*nm_total columns contain dz
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


def forward_obs(data_matrix: np.ndarray,
                coeff: np.ndarray,
                frechxyz: np.ndarray,
                types_sort: np.ndarray
                ) -> (np.ndarray, np.ndarray):
    # TODO: change function to be compatible with no data
    assert len(frechxyz) % 3 == 0, 'frechet matrix incorrect shape'
    # nr of locations
    locs = len(frechxyz) // 3
    times = len(coeff)
    nm_total = len(coeff[0])
    # TODO: pay attention for alternating length data_type
    # has 3 rows (xyz). per row first all times per loc
    # print(frechxyz, coeff)
    xyz = np.matmul(frechxyz, coeff.T).reshape(3, times*locs)
    # print(xyz)
    hor = np.linalg.norm(xyz[:2], axis=0)
    b_int = np.linalg.norm(xyz, axis=0)
    # creates a matrix with shape (7, times * locations)
    # rows mx, my, mz, hor, int, inc, dec. per row first all times per loc
    forwobs_matrix = np.zeros((7, times*locs))
    forwobs_matrix[:3] = xyz
    forwobs_matrix[3] = hor
    forwobs_matrix[4] = b_int
    forwobs_matrix[5] = np.arcsin(xyz[2] / b_int)
    forwobs_matrix[6] = np.arctan2(xyz[1], xyz[0])
    # print(forwobs_matrix)  # seems fine until here

    # per row first all gh then new time then locs
    width = nm_total*times*locs  # row width
    txyz = np.tile(xyz.reshape(3*locs, times), nm_total).reshape(3, width)
    thor = np.tile(hor.reshape(locs, times), nm_total).flatten()
    tb_int = np.tile(b_int.reshape(locs, times), nm_total).flatten()
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
    # print(frech_matrix)
    # now select useful rows by data_type
    # first reshape forwobs_matrix and frech_mat to correspond to datatypes
    # meaning: one row is one datatype of station at every time
    forwobs_matrix = forwobs_matrix.T.reshape(times, 7*locs).T
    frech_matrix = frech_matrix.T.reshape(nm_total*times, 7*locs).T
    # print(frech_matrix)
    # print(types_sort)
    # print(frech_matrix[types_sort])
    forwobs_matrix = forwobs_matrix[types_sort]
    # print(forwobs_matrix)
    # print(data_matrix)
    resid_matrix_t = (data_matrix - forwobs_matrix).T
    # print(resid_matrix_t)
    # print(data_matrix - forwobs_matrix)
    type06 = types_sort % 7
    # inc and dec check
    resid_matrix_t = np.where((type06 == 5) | (type06 == 6), np.arctan2(
        np.sin(resid_matrix_t), np.cos(resid_matrix_t)), resid_matrix_t)

    return frech_matrix[types_sort], resid_matrix_t.T
