import pytest
import numpy as np
from pathlib import Path
import os

from geomagnetic_field_inversions import StationData
from geomagnetic_field_inversions.tools import FieldInversionNoTime
from geomagnetic_field_inversions.forward_modules import fwtools, frechet

SCENARIOS = [(3, [0, 1, 2], 2), (3, [4, 5, 6], 40),
             (10, [0, 1, 2], 2), (10, [4, 5, 6], 40)]


def read_igrf(degree: int) -> (np.ndarray, np.ndarray):
    """ Reads IGRF data"""
    nm_total = (degree+1)**2 - 1
    path = Path(os.path.dirname(__file__))
    f = open(path / 'IGRFcoeff.txt', 'r')

    for r, row in enumerate(f):
        if r == 1:
            time = np.array(row.split()[3:-1], dtype=float)
            igrfcoef = np.zeros((len(time), nm_total))
        if r > 1:
            igrfcoef[:, r-2] = np.array(row.split()[3:-1], dtype=float)
        if r > nm_total:
            break

    return time, igrfcoef


@pytest.mark.parametrize("maxdegree, types, max_iter", SCENARIOS)
def test_model(maxdegree: int,
               types: list,
               max_iter: int,
               latp: int = 15,
               lonp: int = 15,
               t_index: int = -1):
    """ Tests the FieldInversion_notime class

        Parameters
        ----------
        maxdegree
            maximum spherical harmonics degree
        types
            list of integers corresponding to datatypes to be tested.
            [x, y, z, hor, int, incl, decl]
        max_iter
            maximum amount of iterations for inversion
        latp
            number of points in latitudinal direction
        lonp
            number of points in longitudinal direction
        t_index
            which timestep of the IGRF coefficients to use

        If test successful returns: 'test succesfull'
    """
    datatypes = ['x', 'y', 'z', 'hor', 'int', 'inc', 'dec']
    errortypes = [1, 1, 1, 1, 1, np.radians(1), np.radians(1)]
    # load coefficients
    time_array, igrfcoef = read_igrf(maxdegree)
    time = time_array[t_index]
    igrfcoef_time = igrfcoef[t_index]
    forwlat = np.linspace(-80, 80, latp)
    forwlon = np.linspace(-180, 180, lonp)
    longrid, latgrid = np.meshgrid(forwlon, forwlat)
    latgrid = latgrid.flatten()
    longrid = longrid.flatten()
    world_coord = np.zeros((len(latgrid), 3))
    world_coord[:, 0] = np.radians(90-latgrid)
    world_coord[:, 1] = np.radians(longrid)
    world_coord[:, 2] = 6371.2
    # generate linear observations around the globe
    frechxyz = frechet.frechet_basis(world_coord, maxdegree)
    forw_obs = fwtools.forward_obs(igrfcoef, frechxyz)
    forw_obs[5::7] = np.degrees(forw_obs[5::7])
    forw_obs[6::7] = np.degrees(forw_obs[6::7])
    # use class
    testclass = FieldInversionNoTime(time=time, maxdegree=maxdegree,
                                     geodetic=False)
    for i, loc in enumerate(world_coord):
        name = f'{world_coord[i]}'
        temp = StationData(lat=latgrid[i], lon=longrid[i], name=name)
        for typ in types:
            timedata = np.zeros((2, len(time_array)))
            timedata[0] = time_array
            timedata[1] = forw_obs[typ+i*7]
            temp.add_data(datatypes[typ], data=timedata,
                          time_factor=1, error=[errortypes[typ]])
        temp.fitting()
        testclass.add_data(temp)
    testclass.prepare_inversion()
    x0 = np.zeros(testclass._nm_total)
    x0[0] = 30000
    testclass.run_inversion(x0=x0, max_iter=max_iter)
    # compare to initial coeffs
    rel_diff = igrfcoef_time - testclass.unsplined_iter_gh[-1]
    index = np.where(igrfcoef_time != 0)
    rel_diff[index] /= igrfcoef_time[index]
    assert np.all(abs(rel_diff) < 1e-5), 'test failed'
    print('test successful')
