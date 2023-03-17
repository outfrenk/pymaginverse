import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from pathlib import Path
from typing import Union
import pandas as pd


def plot_residuals(ax,
                   invmodel):
    """ Plots the residuals of the geomagnetic field inversion per iteration

    Parameters
    ----------
    ax
        Matplotlib axis object
    invmodel
        An instance of the `geomagnetic_field_inversion` class. This function
        only uses the res_iter attribute.
    """
    im = invmodel
    for i in range(8):
        if im.res_iter[0, i] > 0:
            if i == 0:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_X',
                        linestyle='dotted')
            if i == 1:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_Y',
                        linestyle='dashdot')
            if i == 2:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_Z',
                        linestyle='dashed')
            if i == 3:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_H')
            if i == 4:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_int',
                        linestyle='dotted')
            if i == 5:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_incl',
                        linestyle='dashdot')
            if i == 6:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='black', label='rms_decl',
                        linestyle='dashed')
            if i == 7:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / im.res_iter[0, i],
                        color='red', label='rms_all')
    return ax


def plot_powerspectrum(ax,
                       invmodel,
                       plot_time: int = 'all',
                       plot_iter: int = -1):
    """ Plots the powerspectrum of the gaussian coefficients

    Parameters
    ----------
    ax
        Matplotlib axis object
    invmodel
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter, t_array, _nm_total, and maxdegree attributes.
    plot_time
        Determines which timestep is used to plot powerspectrum. Defaults to
        averaging over all timesteps.
    plot_iter
        Determines which iteration is used to plot powerspectrum. Defaults to
        final iteration.
    """
    im = invmodel
    counter = 0
    sum_coeff = np.zeros(im.maxdegree)
    coeff = (im.unsplined_iter[plot_iter, :] ** 2).reshape(im._nm_total, -1)
    if plot_time == 'all':
        coeff = np.sum(coeff, axis=1) / len(im.t_array)
    else:
        coeff = coeff[:, plot_time]
    for l in range(im.maxdegree):
        for m in range(l+1):
            sum_coeff[l] += coeff[counter]
            counter += 1
    ax.plot(np.arange(1, im.maxdegree+1), sum_coeff, marker='o')
    return ax

def plot_sweep(ax,
               spatial_range: Union[list, np.ndarray],
               temporal_range: Union[list, np.ndarray],
               plot_spatial: bool = True,
               basedir: Union[str, Path] = '.',
               cmap: str = 'RdYlBu'):
    """ Plots a residual-modelsize plot to determine optimal damping parameters
    This function only works after running field_inversion.sweep_damping

    Parameters
    ----------
    ax
        Matplotlib axis object
    spatial_range
        range of spatial damping parameters
    temporal_range
        range of temporal damping parameters
    plot_spatial
        if True, plot spatial damping while temporal damping is static
        if False, plot temporal damping while spatial damping is static
    basedir
        path to coefficients and residuals after each iteration as produced by
        field_inversion.sweep_damping
    cmap
        matplotlib colormap used for plotting

    """
    basedir = Path(basedir)
    basedir.mkdir(exist_ok=True)

    modelsize = np.zeros((len(spatial_range), len(temporal_range)))
    res = np.zeros((len(spatial_range), len(temporal_range)))
    for j, temporal_df in enumerate(temporal_range):
        for i, spatial_df in enumerate(spatial_range):
            if (basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}t_all_coeff.npy'
            ).is_file():
                coef = np.load(basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}'
                                         't_all_coeff.npy')[-1]
            elif (basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}t_final_'
                            'coeff.npy').is_file():
                coef = np.load(basedir / f'{spatial_df:.2e}s+'
                                         '{temporal_df:.2e}t_final_coeff.npy')
            else:
                raise Exception('Could not find file for spatial_df='
                                f'{spatial_df:.2e} and temporal_df='
                                f'{temporal_df:.2e} in {basedir}')
            modelsize[i, j] = np.linalg.norm(coef)
            res[i, j] = pd.read_csv(basedir / f'{spatial_df:.2e}s+'
                                              f'{temporal_df:.2e}t_'
                                              'residual.csv', delimiter=';'
                                    ).to_numpy()[-1, -1]
    if plot_spatial:
        colors = cm.get_cmap(cmap, len(temporal_range))
        for j in range(len(temporal_range)):
            ax.plot(modelsize[:, j], res[:, j], marker='o',
                    color=colors(j / len(temporal_range)))
    else:
        colors = cm.get_cmap(cmap, len(spatial_range))
        for i in range(len(spatial_range)):
            ax.plot(modelsize[i, :], res[i, :], marker='o',
                    color=colors(i / len(spatial_range)))

    return ax
