import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Literal, Tuple
import pandas as pd
import cartopy.crs as ccrs

from .field_inversion import FieldInversion
from .forward_modules import frechet, fwtools
from .data_prep import StationData
from .tools import bsplines

_DataTypes = Literal['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']


def plot_residuals(ax: plt.Axes,
                   im: FieldInversion,
                   **plt_kwargs
                   ) -> plt.Axes:
    """ Plots the residuals of the geomagnetic field inversion per iteration

    Parameters
    ----------
    ax
        Matplotlib axis object
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the res_iter attribute.
    **plt_kwargs
        optional plotting keyword arguments
    """
    lines = ['dotted', 'solid', 'dashdot']
    dt = ['x', 'y', 'z', 'hor', 'inc', 'dec', 'int', 'all']
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative residual')
    count_type = np.zeros(8)
    count_type[:7] = im.count_type
    count_type[7] = sum(im.count_type)

    for i in range(8):
        if count_type[i] > 0:
            if not plt_kwargs:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / max(im.res_iter[:, i]),
                        label=f'rms {dt[i]}', linestyle=lines[i % 3])
            else:
                ax.plot(np.arange(len(im.res_iter)),
                        im.res_iter[:, i] / max(im.res_iter[:, i]),
                        **plt_kwargs)

    return ax


def plot_coeff(ax: plt.Axes,
               im: FieldInversion,
               degree: int = None,
               index: list = None,
               it: int = -1
               ) -> plt.Axes:
    """ Plots Gaussian coefficients through time

    Parameters
    ----------
    ax
        Matplotlib axis object
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter_gh, and t_array attributes.
    degree
        integer of degree of all g's and h's to print.
        If given you do not have to use index.
    index
        List containing the index of the gaussian coefficients to plot,
        assuming ordering like: g^0_1, g^1_1, h^1_1, g^0_2, etc..
    it
        Determines which iteration is used to plot coefficients. Defaults to
        final iteration.
    """
    # TODO: add uncertainty bars
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot',
                  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10)), (0, (1, 10)),
                  (0, (3, 10, 1, 10, 1, 10))]
    markerstyles = ['o', 's', '*', 'D', 'x']
    colorstyles = ['black', 'grey', 'lightgrey']

    if degree is None:
        if index is None:
            raise ValueError('Either degree or index has to be inputted')
        else:
            degree = int(np.sqrt(max(index) + 1))
    else:
        index = np.arange(degree**2, (degree+1)**2) - 1
    labels = []
    for deg in np.arange(1, degree+1):
        labels.append(f'$g^0_{deg}$')
        for m in np.arange(1, deg + 1):
            labels.extend([f'$g^{m}_{deg}$', f'$h^{m}_{deg}$'])

    labels = [labels[i] for i in index]
    for i, item in enumerate(index):
        ax.plot(im.t_array, im.unsplined_iter_gh[it](im.t_array)[:, item],
                linestyle=linestyles[i % len(linestyles)],
                marker=markerstyles[i % len(markerstyles)],
                color=colorstyles[i % len(colorstyles)], label=labels[i])
    return ax


def plot_spectrum(axes: Tuple[plt.Axes, plt.Axes],
                  im: plt.Axes,
                  time: Union[list, np.ndarray] = None,
                  cmb: bool = False
                  ) -> Tuple[plt.Axes, plt.Axes]:
    """ Plots the powerspectrum of gaussian coefficients and its variance at
    the core mantle boundary (cmb) or earth's surface.

    Parameters
    ----------
    axes
        List of matplotlib axis objects
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the splined_gh, t_array, and maxdegree attributes.
    cmb
        If true presents results at cmb, else (default) at earth's surface
    time
        List of indices used to plot powerspectrum. Defaults to using all
        timesteps
    """
    if time is None:
        time = np.arange(im.times - 1)
    coeff = im.splined_gh
    if cmb:
        counter = 0
        for l in range(im.maxdegree):
            mult_factor = (6371.2 / 3485.0) ** (l + 1)
            for m in range(l + 1):
                coeff[:, counter] *= mult_factor
                counter += 1

    spl1 = bsplines.derivatives(im._t_step, 1, derivative=1).flatten()
    spl0 = bsplines.derivatives(im._t_step, 1, derivative=0).flatten()
    coeff_big = np.vstack((np.zeros((im._SPL_DEGREE, im._nm_total)), coeff))
    coeff_sv = np.zeros((len(coeff), im._nm_total))
    coeff_pow = np.zeros((len(coeff), im._nm_total))
    for t in range(len(coeff)):
        # calculate Gauss coefficient according to derivative spline
        coeff_pow[t] = np.matmul(spl0, coeff_big[t:t + im._SPL_DEGREE+1])**2
        coeff_sv[t] = np.matmul(spl1, coeff_big[t:t + im._SPL_DEGREE+1])**2
    coeff_pow = np.sum(coeff_pow[im._SPL_DEGREE-1:], axis=0) / len(time)
    coeff_sv = np.sum(coeff_sv[im._SPL_DEGREE-1:], axis=0) / len(time)
    counter = 0
    sum_coeff_pow = np.zeros(im.maxdegree)
    sum_coeff_sv = np.zeros(im.maxdegree)
    for l in range(im.maxdegree):
        for m in range(2*l + 1):
            sum_coeff_pow[l] += coeff_pow[counter]
            sum_coeff_sv[l] += coeff_sv[counter]
            counter += 1

    axes[0].plot(np.arange(1, im.maxdegree + 1), sum_coeff_pow,
                 marker='o', label='power')
    axes[0].set_xlabel('degree')
    axes[0].set_ylabel('power')
    axes[1].plot(np.arange(1, im.maxdegree + 1), sum_coeff_sv,
                 marker='s', label='variance')
    axes[1].set_xlabel('degree')
    axes[1].set_ylabel('secular variation')

    return axes


def plot_dampnorm(ax: plt.Axes,
                  im: FieldInversion,
                  spatial: bool = True,
                  **plt_kwargs
                  ) -> plt.Axes:
    """
    Plot the Spatial and temporal norm in one figure

    Parameters
    ----------
    ax
        Matplotlib axis object
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the t_array, spat_norm, and temp_norm attribute.
    spatial
        if True plots spatial norm, otherwise temporal norm
    **plt_kwargs
        optional plotting keyword arguments
    """
    ax.set_xlabel('Centre of time interval')
    if spatial:
        ax.plot(im._t_array, im.spat_norm, label='spatial', **plt_kwargs)
        ax.set_ylabel('spatial damping')
    else:
        ax.plot(im._t_array, im.temp_norm, label='temporal', **plt_kwargs)
        ax.set_ylabel('temporal damping')

    return ax


def plot_world(axes: Tuple[plt.Axes, plt.Axes, plt.Axes],
               im: FieldInversion,
               proj: ccrs,
               time: float,
               cmb: bool = False,
               it: int = -1,
               contour_kw: dict = None
               ) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
    """ Plots the magnetic field on Earth given gaussian coefficients

    Parameters
    ----------
    axes
        3 Matplotlib axes objects with appropriate projection
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter_gh and maxdegree attributes.
    proj
        Projection type used for plotting on a world map. Should be an instance
        of cartopy.crs
    time
        Plotting time
    cmb
        Boolean that determines whether to plot the field at the cmb
    it
        Determines which iteration is used to plot powerspectrum. Defaults to
        final iteration.
    contour_kw
        optional plotting parameters
    """
    coeff = im.unsplined_iter_gh[it](time)[np.newaxis, :]
    # make a grid of coordinates and apply forward model
    forwlat = np.arange(-89, 90, 1)
    forwlon = np.arange(0, 360, 1)
    longrid, latgrid = np.meshgrid(forwlon, forwlat)
    latgrid = latgrid.flatten()
    longrid = longrid.flatten()
    world_coord = np.zeros((len(latgrid), 3))
    world_coord[:, 0] = np.radians(90 - latgrid)
    world_coord[:, 1] = np.radians(longrid)
    if cmb:
        world_coord[:, 2] = 3485.0
    else:
        world_coord[:, 2] = 6371.2
    frechxyz = frechet.frechet_basis(world_coord, im.maxdegree)
    forw_obs = fwtools.forward_obs(coeff, frechxyz, reshape=False)
    if cmb:
        plot0 = forw_obs[0]
        title0 = 'North'
        plot1 = forw_obs[1]
        title1 = 'East'
        plot2 = -1 * forw_obs[2]  # Br
        title2 = 'Radial'
    else:
        plot0 = forw_obs[4]
        title0 = 'Intensity'
        plot1 = forw_obs[5]
        title1 = 'Inclination'
        plot2 = forw_obs[6]
        title2 = 'Declination'

    default_kw = {'lvf_0': np.linspace(min(plot0), max(plot0), 10),
                  'lv_0': np.linspace(min(plot0), max(plot0), 10),
                  'cmap_0': 'RdBu_r',
                  'lvf_1': np.linspace(min(plot1), max(plot1), 10),
                  'lv_1': np.linspace(min(plot1), max(plot1), 10),
                  'cmap_1': 'RdBu_r',
                  'lvf_2': np.linspace(min(plot2), max(plot2), 10),
                  'lv_2': np.linspace(min(plot2), max(plot2), 10),
                  'cmap_2': 'RdBu_r'}
    if contour_kw is None:
        contour_kw = default_kw
    else:
        for item in default_kw:
            if item not in contour_kw:
                contour_kw[item] = default_kw[item]

    axes[0].set_global()
    axes[0].contourf(forwlon, forwlat, plot0.reshape(179, 360),
                     cmap=contour_kw['cmap_0'],
                     levels=contour_kw['lvf_0'], transform=proj)
    c = axes[0].contour(forwlon, forwlat, plot0.reshape(179, 360),
                        levels=contour_kw['lv_0'], colors='k',
                        transform=proj)
    axes[0].coastlines()
    axes[0].gridlines()
    axes[0].clabel(c, fontsize=12, inline=True, fmt='%.1f')
    axes[0].set_title(title0)

    axes[1].set_global()
    axes[1].contourf(forwlon, forwlat, plot1.reshape(179, 360),
                     cmap=contour_kw['cmap_1'],
                     levels=contour_kw['lvf_1'], transform=proj)
    c = axes[1].contour(forwlon, forwlat, plot1.reshape(179, 360),
                        levels=contour_kw['lv_1'], colors='k',
                        transform=proj)
    axes[1].coastlines()
    axes[1].gridlines()
    axes[1].clabel(c, fontsize=12, inline=True, fmt='%.1f')
    axes[1].set_title(title1)

    axes[2].set_global()
    axes[2].contourf(forwlon, forwlat, plot2.reshape(179, 360),
                     cmap=contour_kw['cmap_2'],
                     levels=contour_kw['lvf_2'], transform=proj)
    c = axes[2].contour(forwlon, forwlat, plot2.reshape(179, 360),
                        levels=contour_kw['lv_2'], colors='k',
                        transform=proj)
    axes[2].coastlines()
    axes[2].gridlines()
    axes[2].clabel(c, fontsize=12, inline=True, fmt='%.1f')
    axes[2].set_title(title2)
    return axes


def plot_place(ax: plt.Axes,
               im: FieldInversion,
               input_coord: Union[list, np.ndarray],
               datatype: _DataTypes,
               it: int = -1,
               plot_kwargs: dict = None
               ) -> plt.Axes:
    """ Plots the magnetic field on Earth given Gauss coefficients

    Parameters
    ----------
    ax
        Matplotlib axis objects
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter_gh, t_array, and maxdegree attributes.
    input_coord
        3-long array of coordinates containing latitude, longitude, and radius.
    datatype
        type of data to be plotted; should be either: 'x', 'y', 'z', 'hor',
        'inc' (degrees), 'dec' (degrees), or 'int'
    it
        Determines which iteration is used to plot powerspectrum. Defaults to
        final iteration.
    plot_kwargs
        optional plotting keyword arguments
    """
    if plot_kwargs is None:
        plot_kwargs = {'color': 'black', 'linestyle': 'dashed', 'marker': 'o'}
    # translation datatypes
    typedict = {"x": 0, "y": 1, "z": 2, "hor": 3,
                "int": 4, "inc": 5, "dec": 6}
    coeff = im.unsplined_iter_gh[it](im.t_array)
    coord = np.zeros((1, 3))
    coord[0, 0] = 0.5 * np.pi - np.radians(input_coord[0])
    coord[0, 1] = np.radians(input_coord[1])
    if len(input_coord) == 2:
        coord[0, 2] = 6371.2
    else:
        coord[0, 2] = input_coord[2]
    frechxyz = frechet.frechet_basis(coord, im.maxdegree)
    forw_obs = fwtools.forward_obs(coeff, frechxyz)
    forw_obs[5:7] = np.degrees(forw_obs[5:7])
    ax.plot(im.t_array, forw_obs[typedict[datatype]],
            label='model', **plot_kwargs)
    return ax


def compare_loc(axes: list,
                im: FieldInversion,
                dc: StationData,
                plot_fit: bool = False,
                plot_kwargs: dict = None
                ) -> list:
    """Plots the modeled magnetic field at the location of a data input class

    Parameters
    ----------
    axes
        Matplotlib axes objects
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter_gh, t_array, and maxdegree attributes.
    dc
        An instance of the "StationData" class. This function uses the
        lat, loc, types, fit_data, and data attributes of this class
    plot_fit
        If True plots dc.fit_data instead of the original data
    plot_kwargs
        optional plotting keyword arguments

    This function calls plot_place for plotting of the modeled field
    """
    # circumvent length errors
    if len(dc.types) == 1:
        axes = [axes]
    if len(axes) != len(dc.types):
        raise Exception('Not enough axes defined'
                        f', you need {len(dc.types)} axes.')
    # find rejected data
    rej_xdata = find_rejected(dc, im)
    for i, item in enumerate(dc.types):
        xdata = np.array(dc.data[i][0])
        if plot_fit:
            ydata = np.array(dc.fit_data[i](xdata))
        else:
            ydata = np.array(dc.data[i][1])

        if item == 'inc':
            ydata = ydata % 180
            ydata = np.where(ydata > 90, ydata - 180, ydata)
            axes[i].set_ylabel('%s (degrees)' % item)
            axes[i].scatter(xdata, ydata, label='data')
        elif item == 'dec':
            ydata = np.array(ydata) % 360
            ydata = np.where(ydata > 180, ydata - 360, ydata)
            axes[i].set_ylabel('%s (degrees)' % item)
            axes[i].scatter(xdata, ydata, label='data')
        else:
            axes[i].set_ylabel('%s' % item)
            axes[i].scatter(xdata, ydata, label='data')
        # plot rejected data
        for rej in rej_xdata[i]:
            axes[i].axvspan(rej[0], rej[1], alpha=0.5, color='red')
        mindata, maxdata = min(xdata), max(xdata)
        minmodel, maxmodel = min(im.t_array), max(im.t_array)
        axes[i].set_xlabel('Time')
        axes[i] = plot_place(axes[i], im, [dc.lat, dc.lon], item,
                             plot_kwargs=plot_kwargs)
        axes[i].set_xlim(max(mindata, minmodel)*0.9,
                         min(maxdata, maxmodel)*1.1)
        axes[i].legend()
    return axes


def plot_sweep(axes: Tuple[plt.Axes, plt.Axes],
               spatial_range: Union[list, np.ndarray],
               temporal_range: Union[list, np.ndarray],
               basedir: Union[str, Path] = '.',
               ) -> Tuple[plt.Axes, plt.Axes]:
    """ Produces a residual-modelsize plot to determine optimal damp parameters
    This function only works after running field_inversion.sweep_damping

    Parameters
    ----------
    axes
        List of two matplotlib axis objects
    spatial_range
        range of spatial damping parameters
    temporal_range
        range of temporal damping parameters
    basedir
        path to file containing coefficients and residuals after each iteration
        as produced by field_inversion.sweep_damping
    """
    marker = ['o', 'v', '^', 's', '+', '*', 'x', 'd']
    lstyle = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]

    modelsize = np.zeros((len(spatial_range), len(temporal_range)))
    res = np.zeros((len(spatial_range), len(temporal_range)))
    for j, temporal_df in enumerate(temporal_range):
        for i, spatial_df in enumerate(spatial_range):
            coeff = np.load(
                basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}t_final.npy')
            modelsize[i, j] = np.linalg.norm(coeff)
            res[i, j] = pd.read_csv(
                basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}t_residual.csv',
                delimiter=';').to_numpy()[-1, -1]

    tp_ranges = [temporal_range, spatial_range]
    ts = ['t', 's']
    for p in range(2):
        axes[p].set_xlabel('residual')
        axes[p].set_ylabel('model size')
        if p == 1:
            res = res.T
            modelsize = modelsize.T
        for i in range(len(tp_ranges[p])):
            axes[p].plot(res[:, i], modelsize[:, i], color='black',
                         linestyle=lstyle[i % len(lstyle)],
                         label=f'{ts[p]}={tp_ranges[p][i]:.1e}')
        for i in range(len(tp_ranges[p])):
            for j in range(len(tp_ranges[~p])):
                if i == 0:
                    axes[p].scatter(res[j, i], modelsize[j, i], color='black',
                                    marker=marker[j % len(marker)],
                                    label=f'{ts[~p]}={tp_ranges[~p][j]:.1e}')
                else:
                    axes[p].scatter(res[j, i], modelsize[j, i], color='black',
                                    marker=marker[j % len(marker)])
    return axes


def find_rejected(dc: StationData,
                  im: FieldInversion
                  ) -> list:
    """ Find the timespan for which the data is rejected

    Parameters
    ----------
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the station_coord, _t_step, times, types_sorted,
        and accept_matrix attributes.
    dc
        An instance of the "StationData" class. This function uses the lon and
        types attributes of this class
    Returns
    -------
    rej_xdata
        list of lists of lists containing time intervals, per row, at which
        data was rejected. Shape: len(dc.types) X # rejections X 2
    """
    typedict = {"x": 0, "y": 1, "z": 2, "hor": 3,
                "int": 4, "inc": 5, "dec": 6}

    index = np.where(im.station_coord[:, 1] == np.radians(dc.lon))[0]
    if len(index) == 0:
        raise Exception('No match in longitude between classes')
    elif len(index) > 1:
        raise Exception('More than one match in longitude between classes')

    rej_xdata = []
    t_half1 = im._t_step * np.arange(im.times) - 0.25 * im._t_step
    t_half2 = im._t_step * np.arange(im.times) + 0.25 * im._t_step
    for t, types in enumerate(dc.types):
        sorted_row = index * 7 + typedict[types]
        row = np.where(im.types_sorted == sorted_row)[0]
        if len(row) != 1:
            raise Exception('incorrect length row, panic now!')
        rej = np.where(im.accept_matrix[row] == 0)[1]
        rej_list = []
        if len(rej) > 0:
            min_item = rej[0]
            max_item = rej[0]
            for i, item in enumerate(rej[1:]):
                # if in order
                if item == (rej[i] + 1):
                    max_item = item
                else:
                    rej_list.append([t_half1[min_item], t_half2[max_item]])
                    min_item = item
                    max_item = item
                if item == rej[-1]:
                    rej_list.append([t_half1[min_item], t_half2[max_item]])
        rej_xdata.append(rej_list)

    return rej_xdata
