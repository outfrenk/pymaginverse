#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# python setup.py build_ext --inplace

from cython.parallel import prange

import numpy as np


def build_banded(
    double[:, ::1] base_DIF,
    double[:, ::1] temporal,
    int p,
    int[::1] nonzero_inds,
    int[::1] starts,
):
    # Calculate the normal equations matrix by using the precalculated
    # nonzero indices to speed up the loops below
    cdef int n_t = temporal.shape[0]
    cdef int n_coeffs = base_DIF.shape[0]
    cdef int bandw = (p + 1) * n_coeffs + 1
    cdef double[:, ::1] banded = np.zeros(
        (bandw, n_t*n_coeffs),
        dtype=np.float64,
    )
    cdef int k, it_t, it_s, jt_t, jt_s, it, jt, kt, ind


    for it in prange(bandw, nogil=True):
        for jt in range(n_t*n_coeffs-it):
            k = bandw-it-1
            it_t = (it + jt) // n_coeffs
            jt_t = jt // n_coeffs

            it_s = (it + jt) % n_coeffs
            jt_s = jt % n_coeffs
            ind = it_t * n_t + jt_t

            for kt in range(starts[ind], starts[ind+1]):
                banded[k, it + jt] += (
                    temporal[it_t, nonzero_inds[kt]]
                    * base_DIF[it_s, nonzero_inds[kt]]
                    * temporal[jt_t, nonzero_inds[kt]]
                    * base_DIF[jt_s, nonzero_inds[kt]]
                )

    return banded
