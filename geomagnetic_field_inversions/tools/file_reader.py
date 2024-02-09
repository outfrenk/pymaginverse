import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from pathlib import Path
from ..field_inversion import FieldInversion


def read_gauss(coeff: np.ndarray,
               maxdegree: int,
               time_array: np.ndarray,
               splined: bool = True
               ) -> FieldInversion:
    """ Initiates a FieldInversion class based on provided Gauss coefficients

    Parameters
    ----------
    coeff
        Gauss coefficients in a 2D array. Rows correspond to individual time
        steps or knot points. Columns correspond to degree of model.
    maxdegree
        Spherical degree of the Gauss coefficients
    time_array
        Either time array corresponding to rows of coeff (splined = False),
        or knot points corresponding to rows of coeff (splined = True).
    splined
        Whether the Gauss coefficients are in splined form or not.
        Cubic B splines of degree 3 are assumed.

    Returns
    -------
    model
        Instance of the FieldInversion class which only enables the use of
        the following plotting tools:
        1. plot_forward         3. plot_worldmag
        2. plot_coeff           4. plot_cmblontime

    """
    assert len(coeff[0]) == ((maxdegree + 1) ** 2 - 1), \
        f'degree and # gauss coeff ({len(coeff[0])}) do not match'
    if not splined:
        assert len(coeff) == len(time_array), \
            f'length time/knot array and # gauss coeff ({len(coeff)}) do not match'
        model = FieldInversion(t_min=min(time_array), t_max=max(time_array),
                               t_step=time_array[1]-time_array[0],
                               maxdegree=maxdegree)
        model.unsplined_iter_gh = [
            CubicSpline(x=time_array, y=coeff, axis=0, extrapolate=False)]
    else:
        assert len(coeff) == (len(time_array) - 4), \
            f'length time/knot array and # gauss coeff ({len(coeff)}) do not match'
        model = FieldInversion(t_min=min(time_array), t_max=max(time_array),
                               t_step=time_array[1]-time_array[0],
                               maxdegree=maxdegree)
        model.splined_gh = coeff
        model.unsplined_iter_gh = [
            BSpline(t=time_array, c=coeff, k=3, axis=0, extrapolate=False)]
    return model


def read_gaussfile(path: Path,
                   datafile: str
                   ) -> FieldInversion:
    """ Reads Gauss coefficient files as produced by the old Fortran routine

    Parameters
    ----------
    path
        path to file
    datafile
        name of file

    Returns
    -------
     model
        Instance of the FieldInversion class which only enables the use of
        the following plotting tools:
        1. plot_forward         3. plot_worldmag
        2. plot_coeff           4. plot_cmblontime
    """
    f = open(path / datafile, 'r')
    for r, row in enumerate(f):
        if r == 1:
            maxdegree = int(row.split()[0])
            nm_total = (maxdegree+1)**2 - 1
        elif r == 2:
            time_knots = np.array(row.split()).astype(float)
        elif r == 3:
            coeff = np.array(row.split()).astype(float)
            coeff = coeff.reshape((len(time_knots) - 4), nm_total)
            break
    model = read_gauss(coeff, maxdegree, time_knots, splined=True)
    return model
