import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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

    Returns
    -------

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
