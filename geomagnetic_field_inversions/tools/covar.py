import numpy as np
import scipy.linalg as scl
from pathlib import Path


def get_covar(path: Path,
              one_time: bool = False):
    """ Function to calculate and save covariance matrix

    Parameters
    ----------
    path
        path to location of normal_equation matrix + where to save covariance
    one_time
        whether one timestep was used or multiple
    """
    if one_time:
        normal_eq = np.load(path / 'normal_eq.npy')
    else:
        normal_eq = np.load(path / 'normal_eq_splined.npy')
        normal_eq += np.load(path / 'sparse_damp.npy')
    covar = scl.pinv(normal_eq)
    np.save(path / 'covar', covar)
