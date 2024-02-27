#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

# python setup.py build_ext --inplace

from cython.parallel import prange
from cython import cdivision
import numpy as np

@cdivision(True)
def build_banded(double[:, ::1] base_DIF, double[:, ::1] temporal, int p):
    cdef int n_t = temporal.shape[0]
    cdef int n_coeffs = base_DIF.shape[0]
    cdef int bandw = (p + 1) * n_coeffs + 1
    cdef int n_data = base_DIF.shape[1]
    cdef double[:, :] banded = np.zeros((bandw, n_t*n_coeffs))
    cdef int k, it_t, it_s, jt_t, jt_s, it, jt, kt

    for it in prange(bandw, nogil=True):
        k = bandw-1-it
        # XXX This is a bit lazy...I guess thinking about this loops
        # offers the potential for another speed gain.
        for jt in range(n_t*n_coeffs-it):
            it_t = (it + jt) // n_coeffs
            jt_t = jt // n_coeffs

            if p < abs(it_t - jt_t):
                continue

            it_s = (it + jt) % n_coeffs
            jt_s = jt % n_coeffs

            for kt in range(n_data):
                banded[k, it+jt] += (
                    temporal[it_t, kt]
                    * base_DIF[it_s, kt]
                    * temporal[jt_t, kt]
                    * base_DIF[jt_s, kt]
                )

    return banded


@cdivision(True)
def build_banded_2(
    double[:, ::1] base_DIF,
    double[:, ::1] temporal,
    int p,
    long long[::1] nonzero_inds,
    long long[::1] starts,
):
    # Calculate the normal equations matrix by using the precalculated
    # nonzero indices to speed up the loops below
    cdef int n_t = temporal.shape[0]
    cdef int n_coeffs = base_DIF.shape[0]
    cdef int bandw = (p + 1) * n_coeffs + 1
    cdef int n_data = base_DIF.shape[1]
    cdef double[:, :] banded = np.zeros((bandw, n_t*n_coeffs))
    cdef int k, it_t, it_s, jt_t, jt_s, it, jt, kt, ind

    for it in prange(bandw, nogil=True):
        k = bandw-1-it
        for jt in range(n_t*n_coeffs-it):
            it_t = (it + jt) // n_coeffs
            jt_t = jt // n_coeffs

            if p < abs(it_t - jt_t):
                continue

            it_s = (it + jt) % n_coeffs
            jt_s = jt % n_coeffs

            ind = it_t * n_t + jt_t
            for kt in range(starts[ind], starts[ind+1]):
                banded[k, it+jt] += (
                    temporal[it_t, nonzero_inds[kt]]
                    * base_DIF[it_s, nonzero_inds[kt]]
                    * temporal[jt_t, nonzero_inds[kt]]
                    * base_DIF[jt_s, nonzero_inds[kt]]
                )

    return banded
