import numpy as np


def forward_obs(coeff: np.ndarray,
                frechxyz: np.ndarray,
                reshape=True
                ) -> np.ndarray:
    """
    Calculates modeled observations at given locations

    Parameters
    ----------
    coeff
        Gauss coefficients. Each row contains the coefficients of one timestep
    frechxyz
        Frechet matrix for dx, dy, and dz components
    reshape
        If True restructures forward observations in such a way that the first
        set of 7 rows corresponds to x, y, z, h, int, inc, and dec of one
        station. Each row contains the specific datatype at every timestep. The
        next station would have the next 7 rows.
        If False does not restructure forward observations, but leaves it in 7
        rows (x, y, z, h, int, inc, and dec). Where each row contains the data
        of one station at every timestep, whereafter data starts of another
        station in the same way.

    Returns
    -------
    forward_obs
        Forward observations; see reshape parameter for discussion on shape
    """
    assert len(frechxyz) % 3 == 0, 'frechet matrix incorrect shape'
    assert coeff.ndim == 2, 'Gauss coefficients have incorrect dimensions'
    # nr of locations
    locs = len(frechxyz) // 3
    times = len(coeff)
    # has 3 rows (xyz). per row first all times per loc
    xyz = np.matmul(frechxyz, coeff.T).reshape(3, times*locs)
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

    if reshape:
        forwobs_matrix = forwobs_matrix.reshape(
            7, locs, times).swapaxes(0, 1).reshape(7*locs, times)
    return forwobs_matrix


def residual_obs(forwobs_matrix: np.ndarray,
                 data_matrix: np.ndarray,
                 types_sort: np.ndarray
                 ) -> np.ndarray:
    """
    Calculates the residual by subtracting forward observation from data.
    Applies angular correction to decl/incl data

    Parameters
    ----------
    forwobs_matrix
        Contains the modeled observations
    data_matrix
        Contains the real observations (from the data)
    types_sort
        Tells the type of data by providing an index to every row in either
        forwobs_matrix or data_matrix

    Returns
    -------
    resid_matrix_t.T
        Residual; size is similar to forwobs_matrix or data_matrix
    """
    assert forwobs_matrix.shape == data_matrix.shape, 'shapes are not similar'
    resid_matrix_t = (data_matrix - forwobs_matrix).T
    type06 = types_sort % 7
    # inc and dec check
    resid_matrix_t = np.where((type06 == 5) | (type06 == 6), np.arctan2(
        np.sin(resid_matrix_t), np.cos(resid_matrix_t)), resid_matrix_t)
    return resid_matrix_t.T


def residual_type(residual_weighted: np.ndarray,
                  types_sort: np.ndarray,
                  count_type: np.ndarray
                  ) -> np.ndarray:
    """
    Calculates the RMS per datatype and sums them

    Parameters
    ----------
    residual_weighted
        residuals weighted by expected error
    types_sort
        Contains datatype number referring to rows in residual_weighted
    count_type
        Contains the occurence of the 7 datatypes provided by a station
        summed over time

    Returns
    -------
    res_iter
        RMS weighted residual per datatype. First 7 items correspond to
        possible presence of x, y, z, h, int, inc, or dec. Item 8 is the sum
    """
    type06 = types_sort % 7
    res_iter = np.zeros(8)
    for i in range(7):
        if count_type[i] != 0:
            res_iter[i] = np.sqrt(np.sum(
                residual_weighted[type06 == i]**2) / count_type[i])
    res_iter[7] = np.sqrt(np.sum(residual_weighted**2) / sum(count_type))
    return res_iter
