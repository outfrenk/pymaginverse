import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Literal, Tuple
import pandas as pd
import cartopy.crs as ccrs

from .field_inversion import FieldInversion
from .forward_modules import frechet, fwtools
from .data_prep import StationData
from .tools import bsplines, FieldInversionNoTime

_DataTypes = Literal['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']


def plot_data(axes: Union[list, plt.Axes],
              dc: StationData,
              ) -> plt.Axes:
    """ Plots input (paleo)magnetic data based on the StationData class

    Parameters
    ----------
    axes
        List of matplotlib axis objects equal to dc.types, or one axes
    dc
        An instance of the "StationData" class. This function uses the
        lat, loc, types, and data attributes of this class
    """
    if len(dc.types) == 1 and type(axes) != list:
        axes = [axes]
    assert len(axes) >= len(dc.types), 'not defined enough plot axes'
    for i in range(len(dc.types)):
        axes[i].set_title('Fitting %s of data' % dc.types[i])
        if dc.types[i] == 'inc' or dc.types[i] == 'dec':
            axes[i].set_ylabel('%s (degrees)' % dc.types[i])
        else:
            axes[i].set_ylabel('%s' % dc.types[i])
        axes[i].set_xlabel('Time')
        axes[i].scatter(dc.data[i][0], dc.data[i][1], label='data')
        axes[i].legend()
    return axes


def compare_loc(axes: list,
                im: FieldInversion,
                dc: StationData,
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
    plot_kwargs
        optional plotting keyword arguments

    This function calls plot_place for plotting of the modeled field
    """
    # circumvent length errors
    if len(dc.types) == 1 and type(axes) != list:
        axes = [axes]
    if len(axes) != len(dc.types):
        raise Exception('Not enough axes defined'
                        f', you need {len(dc.types)} axes.')

    for i, item in enumerate(dc.types):
        xdata = np.array(dc.data[i][0])
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

        mindata, maxdata = min(xdata), max(xdata)
        minmodel, maxmodel = min(im.t_array), max(im.t_array)
        axes[i].set_xlabel('Time')
        axes[i] = plot_forward(axes[i], im, [dc.lat, dc.lon], item,
                               plot_kwargs=plot_kwargs)
        axes[i].set_xlim(max(mindata, minmodel)*0.9,
                         min(maxdata, maxmodel)*1.1)
        axes[i].legend()
    return axes


def plot_forward(ax: plt.Axes,
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
    forw_obs = fwtools.forward_obs(coeff, frechxyz, link=np.zeros(
        len(im.t_array), dtype=np.int8))
    forw_obs[5:7] = np.degrees(forw_obs[5:7])
    ax.plot(im.t_array, forw_obs[typedict[datatype]],
            label='model', **plot_kwargs)
    return ax


def plot_residuals(ax: plt.Axes,
                   im: Union[FieldInversion, FieldInversionNoTime],
                   **plt_kwargs
                   ) -> plt.Axes:
    """ Plots the residuals of the geomagnetic field inversion per iteration

    Parameters
    ----------
    ax
        Matplotlib axis object
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the res_iter and count_type attribute.
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
               im: Union[FieldInversion, FieldInversionNoTime],
               degree: int = None,
               index: list = None,
               it_time: int = -1,
               std: np.ndarray = None,
               plot_iter: bool = False,
               ) -> plt.Axes:
    """ Plots Gaussian coefficients through time or iteration.
    Note: FieldInversionNoTime only works with plot_iter=True

    Parameters
    ----------
    ax
        Matplotlib axis object
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the unsplined_iter_gh, and t_array attributes.
    degree
        Integer of degree of all g's and h's to print.
        If given you do not have to use index.
    index
        List containing the index of the gaussian coefficients to plot,
        assuming ordering like: g^0_1, g^1_1, h^1_1, g^0_2, etc..
    it_time
        Determines which iteration is used to plot coefficients. Defaults to
        final iteration. if plot_iter is True, it_time indicates which timestep
        to plot. Defaults to final time.
    std
        Array containing the standard deviations of the Gauss coefficients. As
        produced by stdev.py.
    plot_iter
        Boolean indicating whether to plot Gauss coefficients against inversion
        iterations (True) or against time for the chosen iteration (False).
        Default option is set to False.
    """
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
        if plot_iter:
            coeff = np.zeros(len(im.unsplined_iter_gh) + 1)
            if isinstance(im, FieldInversion):
                coeff[0] = im.x0[item]
                for c in range(len(coeff)-1):
                    coeff[c+1] = im.unsplined_iter_gh[c](
                        im.t_array[it_time])[item]
            elif isinstance(im, FieldInversionNoTime):
                coeff[0] = im.x0[item]
                coeff[1:] = im.unsplined_iter_gh[:, item]
            else:
                raise Exception('Class not found')
            ax.plot(np.arange(len(coeff)), coeff,
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markerstyles[i % len(markerstyles)],
                    color=colorstyles[i % len(colorstyles)],
                    markersize=3, label=labels[i])
        else:
            if std is not None:
                ax.errorbar(im.t_array,
                            im.unsplined_iter_gh[it_time](im.t_array)[:, item],
                            yerr=std[:, item], capsize=4,
                            linestyle=linestyles[i % len(linestyles)],
                            color=colorstyles[i % len(colorstyles)],)
            ax.plot(im.t_array,
                    im.unsplined_iter_gh[it_time](im.t_array)[:, item],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markerstyles[i % len(markerstyles)],
                    color=colorstyles[i % len(colorstyles)],
                    markersize=3, label=labels[i])

    return ax


def plot_spectrum(axes: Tuple[plt.Axes, plt.Axes],
                  im: plt.Axes,
                  time: Union[list, np.ndarray] = None,
                  cmb: bool = True
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
        If True (default) presents results at cmb, else at earth's surface
    time
        List of indices used to plot powerspectrum. Defaults to using all
        timesteps
    """
    if time is None:
        time = np.arange(im.times - 1)
    coeff = im.splined_gh
    if cmb:
        depth = 6371.2 / 3485.0
    else:
        depth = 1

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
            sum_coeff_pow[l] += coeff_pow[counter] * (l+2) * depth**(2*(l+1)+4)
            sum_coeff_sv[l] += coeff_sv[counter] * (l+2) * depth**(2*(l+1)+4)
            counter += 1

    axes[0].plot(np.arange(1, im.maxdegree + 1), sum_coeff_pow,
                 marker='o', label='power')
    axes[0].set_xlabel('degree')
    axes[0].set_ylabel('power')
    axes[0].set_yscale('log')
    axes[1].plot(np.arange(1, im.maxdegree + 1), sum_coeff_sv,
                 marker='s', label='variance')
    axes[1].set_xlabel('degree')
    axes[1].set_ylabel('secular variation')
    axes[1].set_yscale('log')

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


def plot_worldmag(axes: Tuple[plt.Axes, plt.Axes, plt.Axes],
                  im: Union[FieldInversion, FieldInversionNoTime],
                  proj: ccrs,
                  plot_loc: bool = False,
                  time: float = None,
                  cmb: bool = False,
                  it: int = -1,
                  contour_kw: dict = None,
                  plotloc_kw: dict = None
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
    plot_loc
        Boolean indicating whether to plot the location of the datapoints
        on the worldmap. Defaults to False.
    time
        Plotting time. If time is None, assuming im is an instance of
        field_inversion_onestep.FieldInversion_notime
    cmb
        Boolean that determines whether to plot the field at the cmb
    it
        Determines which iteration is used to plot powerspectrum. Defaults to
        final iteration.
    contour_kw
        optional plotting parameters
    plotloc_kw
        optional plotting parameters for optional plotting locations on map
    """
    if isinstance(im, FieldInversionNoTime):
        coeff = im.unsplined_iter_gh[it][np.newaxis, :]
    else:
        coeff = im.unsplined_iter_gh[it](time)[np.newaxis, :]
    # make a grid of coordinates and apply forward model
    forwlat = np.arange(-89, 89, 1)
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
    forw_obs = fwtools.forward_obs(coeff, frechxyz, link=np.zeros(
        len(frechxyz), dtype=np.int8))
    plot = np.zeros((3, len(forw_obs[0])))
    title = []
    if cmb:
        plot[0] = forw_obs[0]
        title.append('North')
        plot[1] = forw_obs[1]
        title.append('East')
        plot[2] = -1 * forw_obs[2]  # Br
        title.append('Radial')
    else:
        plot[0] = forw_obs[4]
        title.append('Intensity')
        plot[1] = np.degrees(forw_obs[5])
        title.append('Inclination')
        plot[2] = np.degrees(forw_obs[6])
        title.append('Declination')

    default_kw = {'lvf_0': np.linspace(min(plot[0]), max(plot[0]), 10),
                  'lv_0': np.linspace(min(plot[0]), max(plot[0]), 10),
                  'cmap_0': 'RdBu_r',
                  'lvf_1': np.linspace(min(plot[1]), max(plot[1]), 10),
                  'lv_1': np.linspace(min(plot[1]), max(plot[1]), 10),
                  'cmap_1': 'RdBu_r',
                  'lvf_2': np.linspace(min(plot[2]), max(plot[2]), 10),
                  'lv_2': np.linspace(min(plot[2]), max(plot[2]), 10),
                  'cmap_2': 'RdBu_r'}
    if contour_kw is None:
        contour_kw = default_kw
    else:
        for item in default_kw:
            if item not in contour_kw:
                contour_kw[item] = default_kw[item]

    for i in range(3):
        axes[i].set_global()
        axes[i].contourf(forwlon, forwlat, plot[i].reshape(178, 360),
                         cmap=contour_kw[f'cmap_{i}'],
                         levels=contour_kw[f'lvf_{i}'], transform=proj)
        c = axes[i].contour(forwlon, forwlat, plot[i].reshape(178, 360),
                            levels=contour_kw[f'lv_{i}'], colors='k',
                            transform=proj)
        axes[i].coastlines()
        axes[i].gridlines()
        axes[i].clabel(c, fontsize=12, inline=True, fmt='%.1f')
        axes[i].set_title(title[i])
        if plot_loc:
            axes[i] = plot_worldloc(axes[i], im, proj, plotloc_kw)

    return axes


def plot_worldloc(ax: plt.Axes,
                  im: Union[FieldInversion, FieldInversionNoTime],
                  proj: ccrs,
                  plot_kw: dict = None
                  ) -> plt.Axes:
    """ Plots datum locations on a world map

    Parameters
    ----------
    ax
        Matplotlib axis with appropriate projection
    im
        An instance of the `geomagnetic_field_inversion` class. This function
        uses the station_coord attributes.
    proj
        Projection type used for plotting on a world map. Should be an instance
        of cartopy.crs
    plot_kw
        optional plotting parameters for optional plotting locations on map
    """
    lat = np.degrees(0.5 * np.pi - im.station_coord[:, 0])
    lon = np.degrees(im.station_coord[:, 1])
    if plot_kw:
        ax.scatter(lat, lon, transform=proj, **plot_kw)
    else:
        ax.scatter(lat, lon, transform=proj, color='black', marker='^')
    return ax


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

    spatnorm = np.zeros((len(spatial_range), len(temporal_range)))
    tempnorm = np.zeros((len(spatial_range), len(temporal_range)))
    res = np.zeros((len(spatial_range), len(temporal_range)))
    for j, temporal_df in enumerate(temporal_range):
        for i, spatial_df in enumerate(spatial_range):
            filename = f'{spatial_df:.2e}s+{temporal_df:.2e}t_damp.npy'
            with open(basedir / filename, 'rb') as f:
                spatnorm[i, j] = np.linalg.norm(np.load(f))
                tempnorm[i, j] = np.linalg.norm(np.load(f))
            res[i, j] = pd.read_csv(
                basedir / f'{spatial_df:.2e}s+{temporal_df:.2e}t_residual.csv',
                delimiter=';').to_numpy()[-1, -1]

    tp_ranges = [temporal_range, spatial_range]
    ts = ['t', 's']
    for p in range(2):
        axes[p].set_xlabel('residual')
        axes[p].set_ylabel('damping norm')
        modelsize = spatnorm
        if p == 1:
            res = res.T
            modelsize = tempnorm
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
