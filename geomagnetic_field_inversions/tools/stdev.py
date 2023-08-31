import numpy as np
import scipy.sparse as scs
from pathlib import Path

from . import bsplines


def get_stdev(path: Path,
              degree: int,
              save_covar: bool = False,
              save_res: bool = False,
              one_time: bool = False):
    """ Function to calculate and save standard deviation

    Parameters
    ----------
    path
        path to location of normal_equation matrix + where to save covariance
    degree
        degree of the spherical model.
    save_covar
        boolean indicating whether to save covariance matrix. default is false
    save_res
        boolean indicating whether to save resolution matrix. default is false
    one_time
        whether the class for one timestep, FieldInversion_notime, was used or
        not (FieldInversion)
    """
    spl_degree = 3
    if one_time:
        normal_eq = np.load(path / 'normal_eq.npy')
    else:
        normal_eq = scs.load_npz(path / 'forward_matrix.npz')
        damp = scs.load_npz(path / 'sparse_damp.npz')
        if save_res:
            print('Calculating resolution matrix')
            res_mat = np.linalg.solve((normal_eq+damp).todense(),
                                      normal_eq.todense())
            np.save(path / 'resolution_mat', res_mat)
        normal_eq += damp
    print('start inversion')
    covar = np.linalg.inv(normal_eq.todense())
    print('finished inversion, calculating std')
    nm_total = (degree+1)**2 - 1
    covar_spl = np.sqrt(np.diag(covar)).reshape(-1, nm_total)
    coeff_big = np.vstack((np.zeros((spl_degree, nm_total)), covar_spl))
    std = np.zeros((len(covar_spl), nm_total))
    spl0 = bsplines.derivatives(1, 1, derivative=0).flatten()
    for t in range(len(covar_spl)):
        std[t] = np.matmul(spl0, coeff_big[t:t + spl_degree + 1])
    print('start saving')
    np.save(path / 'std', std[spl_degree-1:])
    if save_covar:
        np.save(path / 'covar', covar)
    print('saving finished')
