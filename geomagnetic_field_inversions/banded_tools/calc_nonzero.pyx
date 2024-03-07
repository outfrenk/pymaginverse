#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

# python setup.py build_ext --inplace

from libcpp.list cimport list as cpplist
cimport openmp

from cython.parallel import prange
from cython import cdivision

import numpy as np

@cdivision(True)
def calc_nonzero(double[:, ::1] temporal):
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    cdef int n_t = temporal.shape[0]
    cdef int n_data = temporal.shape[1]
    cdef cpplist[unsigned int] indices
    cdef long[:] starts = np.zeros(n_t*n_t + 1, dtype=int)
    cdef unsigned int it, it_t, jt_t, kt
    cdef int count
    starts[0] = 0

    for it in range(n_t*n_t):
        it_t = it % n_t
        jt_t = it // n_t
        for kt in range(n_data):
            if temporal[it_t, kt] * temporal[jt_t, kt] != 0:
                openmp.omp_set_lock(&lock)
                indices.push_back(kt)
                starts[it + 1] += 1
                openmp.omp_unset_lock(&lock)

    return starts, indices


@cdivision(True)
def calc_nonzero_parallel(double[:, ::1] temporal):
    cdef int n_t = temporal.shape[0]
    cdef int n_data = temporal.shape[1]
    cdef cpplist[unsigned int] indices
    cdef long[:] starts = np.zeros(n_t*n_t + 1, dtype=int)
    cdef unsigned int it, it_t, jt_t, kt
    cdef int count
    starts[0] = 0

    for it in prange(n_t*n_t, nogil=True):
        it_t = it % n_t
        jt_t = it // n_t
        for kt in range(n_data):
            if temporal[it_t, kt] * temporal[jt_t, kt] != 0:
                indices.push_back(kt)
                starts[it + 1] += 1

    return starts, indices
