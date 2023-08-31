import pytest
import numpy as np
from pathlib import Path

from ..field_inversion import FieldInversion
from ..forward_modules import fwtools, frechet

# load coefficients
# generate linear observations around the globe
# use program
# compare to initial coeffs
# @pytest.mark.parametrize("limit", LIMIT_params, ids=['dip', 'quad', 'oct'])
def read_igrf(degree=10):
    nm_total = (degree+1)**2 - 1
    f = open('IGRFcoeff.txt', 'r')
    igrfcoef = np.zeros(nm_total)
    for r, row in enumerate(f):
        if r > 1:
            igrfcoef[r-2] = float(row.split()[-2])
        if r > nm_total:
            break
    return igrfcoef


def test_linearmodel(max_degree, latp=120, lonp=120):
    igrfcoef = read_igrf(max_degree)
    forwlat = np.linspace(-89, 90, latp)
    forwlon = np.linspace(0, 360, lonp)
    longrid, latgrid = np.meshgrid(forwlon, forwlat)
    latgrid = latgrid.flatten()
    longrid = longrid.flatten()
    world_coord = np.zeros((len(latgrid), 2))
    world_coord[:, 0] = latgrid
    world_coord[:, 1] = longrid
    frechxyz = frechet.frechet_basis(world_coord, max_degree)
    forw_obs = fwtools.forward_obs(igrfcoef, frechxyz, reshape=False)


if __name__ == '__main__':
    # run the test functions
    test_linearmodel()
