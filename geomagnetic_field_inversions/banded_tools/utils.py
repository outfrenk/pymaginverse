import numpy as np
from tqdm import tqdm


def banded_to_full(banded, quiet=True):
    """ Utility routine to transform a banded matrix into a full one """
    n = banded.shape[0]-1
    full = np.zeros(
        (banded.shape[1], banded.shape[1]),
        dtype=banded.dtype,
    )
    for it in tqdm(range(n), disable=quiet):
        full += np.diag(
            banded[it, n-it:],
            k=n-it,
        )
    full += full.T + np.diag(banded[-1])
    return full


def banded_mul_vec(banded, vec, quiet=True):
    """ Utility routine to multpiply a banded matrix and a vector """
    # This could be moved to cython for a slight speedup
    n = banded.shape[0]
    result = banded[-1] * vec
    for it in tqdm(np.arange(n-1)+1, disable=quiet):
        k = banded.shape[0]-1-it
        result[:banded.shape[1]-it] += banded[k, it:] * vec[it:]
        result[it:] += banded[k, it:] * vec[:banded.shape[1]-it]

    return result


def banded_mul_mat(banded, mat, quiet=True):
    """ Utility routine to multpiply a banded matrix and a matrix """
    # n = banded.shape[0]
    # result = banded[-1, :, None] * mat
    # for it in tqdm(np.arange(n-1)+1, disable=quiet):
    #     k = banded.shape[0]-1-it
    #     result[:banded.shape[1]-it] += banded[k, it:, None] * mat[it:]
    #     result[it:] += banded[k, it:, None] * mat[:banded.shape[1]-it]
    result = np.zeros((banded.shape[1], mat.shape[1]))
    for it in tqdm(np.arange(mat.shape[1]), disable=quiet):
        result[:, it] = banded_mul_vec(banded, mat[:, it])

    return result


if __name__ == '__main__':
    rng = np.random.default_rng(1412)

    banded = np.zeros((23, 50))
    for it in range(banded.shape[0]):
        k = banded.shape[0]-1-it
        banded[k, it:] = rng.random(size=banded.shape[1]-it)

    vec = rng.random(size=banded.shape[1])

    full = banded_to_full(banded)
    print(
        np.allclose(
            full @ vec,
            banded_mul_vec(banded, vec),
        )
    )

    mat = rng.random(size=(banded.shape[1], 230))
    print(
        np.allclose(
            full @ mat,
            banded_mul_mat(banded, mat),
        )
    )
