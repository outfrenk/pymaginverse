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


def forward_obs_time(coeff: np.ndarray,
                     frechxyz: np.ndarray,
                     splinebase: np.ndarray,
                     ) -> np.ndarray:
    """
    Calculates modeled observations at given locations

    Parameters
    ----------
    coeff
        Gauss coefficients in a shape corresponding to frechxyz.shape[2],
        splinebase.shape[1]
    frechxyz
        Frechet matrix for dx, dy, and dz components (stations X 3 X nm_total)
    splinebase
        Matrix of spline basis functions

    Returns
    -------
    forward_obs
        Forward observations; each row contains one type
    """
    # TODO assertions for splinebase and documentation

    # XXX: maybe this is slowed down due to transpose etc.
    xyz = np.einsum(
        'ij, klj, ik -> lk',
        coeff,
        frechxyz,
        splinebase,
        optimize=True,
    )
    forwobs_matrix = np.zeros((7, xyz.shape[1]))
    forwobs_matrix[:3] = xyz
    hor = np.linalg.norm(xyz[:2], axis=0)
    b_int = np.linalg.norm(xyz, axis=0)
    forwobs_matrix[3] = hor
    forwobs_matrix[4] = b_int
    forwobs_matrix[5] = np.arcsin(xyz[2] / b_int)
    forwobs_matrix[6] = np.arctan2(xyz[1], xyz[0])

    return forwobs_matrix


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


def calc_forw(maxdegree: int,
              coord: np.ndarray,
              coeff: np.ndarray,
              link: np.ndarray = None):
    """ Calculate the modeled geomagnetic field

    Parameters
    ----------
    maxdegree
        Spherical degree of modeled field
    coord
        coordinates of the locations where to calculate field
    coeff
        Gauss coefficients, each row containing the Gauss coefficients per
        spline
    link
        array that indicates how coordinates are linked to coeff

    Returns
    -------
    forw_obs
        calculated field observations
    """
    frechxyz = frechet_basis(coord, maxdegree)
    forw_obs = forward_obs(coeff, frechxyz, link=link)
    forw_obs[5:7] = np.degrees(forw_obs[5:7])
    return forw_obs
