import numpy as np


def forward_obs(coeff: np.ndarray,
                frechxyz: np.ndarray,
                link: np.ndarray = None,
                ) -> np.ndarray:
    """
    Calculates modeled observations at given locations

    Parameters
    ----------
    coeff
        Gauss coefficients. Each row contains the coefficients of one datum
    frechxyz
        Frechet matrix for dx, dy, and dz components (stations X 3 X nm_total)
    link
        Array which tells how coeff and frechxyz should be combined. If it has
        the same length as coeff if tells to which 1st dimension of frechxyz
        each row of coeff belongs to and vice versa. Ignored if set to None

    Returns
    -------
    forward_obs
        Forward observations; each row contains one type
    """
    assert frechxyz.ndim == 3, 'frechet matrix should have 3 dimensions'
    assert len(frechxyz[0]) == 3, 'frechet matrix does not contain dx, dy, dz'
    assert coeff.ndim == 2, 'Gauss coefficients have incorrect dimensions'
    if link is not None:
        # if only one set of Gaussian coefficients for all points
        if len(coeff) == len(link):
            frechxyz = frechxyz[link]
        elif len(frechxyz) == len(link):
            coeff = coeff[link]
        else:
            raise Exception(f'Link has incorrect size ({len(link)}, should be:'
                            f' {len(coeff)}, {len(frechxyz)}, or None')
    assert len(coeff) == len(frechxyz), 'coeff and frechet unequal # of rows:'\
                                        f' {len(coeff)} vs {len(frechxyz)}'
    # nr of locations
    datums = len(coeff)
    # creates a matrix with shape (7, times * locations)
    # rows mx, my, mz, hor, int, inc, dec. per row first all times per loc
    forwobs_matrix = np.zeros((7, datums))
    forwobs_matrix[0] = np.sum((frechxyz[:, 0] * coeff), axis=1)
    forwobs_matrix[1] = np.sum((frechxyz[:, 1] * coeff), axis=1)
    forwobs_matrix[2] = np.sum((frechxyz[:, 2] * coeff), axis=1)
    xyz = forwobs_matrix[:3]
    hor = np.linalg.norm(xyz[:2], axis=0)
    b_int = np.linalg.norm(xyz, axis=0)
    forwobs_matrix[3] = hor
    forwobs_matrix[4] = b_int
    forwobs_matrix[5] = np.arcsin(xyz[2] / b_int)
    forwobs_matrix[6] = np.arctan2(xyz[1], xyz[0])

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
        Tells the type of data by providing an index to every row in both
        forwobs_matrix and data_matrix

    Returns
    -------
    resid_matrix_t.T
        Residual; size is similar to forwobs_matrix and data_matrix
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
