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
    cdef unsigned int it, jt, kt
    cdef int count
    starts[0] = 0

    for it in prange(n_t, nogil=True):
        for jt in range(n_t):
            for kt in range(n_data):
                if temporal[it, kt] * temporal[jt, kt] != 0:
                    openmp.omp_set_lock(&lock)
                    indices.push_back(kt)
                    openmp.omp_unset_lock(&lock)
                    starts[it * n_t + jt + 1] += 1

    return starts, indices
