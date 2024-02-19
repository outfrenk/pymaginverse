import unittest
from pathlib import Path

import numpy as np
from pandas import read_csv

from scipy.interpolate import BSpline

from geomagnetic_field_inversions.forward_modules.frechet import frechet_basis
from geomagnetic_field_inversions.forward_modules.fwtools import (
    forward_obs,
    forward_obs_time,
)

path = Path(__file__).parent

with open(path / '../benchmark_data/newmod', 'r') as fh:
    t_min, t_max = map(float, fh.readline().split()[0:2])
    # read the rest as a long string
    input_string = ' '.join(fh.readlines())
    # create an array from the string
    input_array = np.fromstring(input_string, dtype=float, sep=' ')
    # get the relevant quantities from the array
    l_max, n_spl = map(int, input_array[:2])
    # read the appropriate number of knots
    knots = input_array[2:n_spl+4+2]
    # read the appropriate number of coefficients
    n_coeffs = l_max * (l_max + 2)
    ref_coeffs = input_array[n_spl+4+2:n_spl*(n_coeffs+1)+4+2]
    ref_coeffs = ref_coeffs.reshape(n_spl, n_coeffs)

ref_spline = BSpline(
    knots,
    ref_coeffs,
    3,
)

raw_data = read_csv(path / '../benchmark_data/testdata.csv')
raw_data = raw_data.query(f'{t_min} <= t <= {t_max}')
raw_data.reset_index(inplace=True, drop=True)
raw_data = raw_data.sample(n=100, random_state=14467)

coords = np.array(
    [
        np.deg2rad(raw_data['colat']),
        np.deg2rad(raw_data['lon']),
        raw_data['rad']
    ],
).T
times = raw_data['t']


class Test_forward(unittest.TestCase):
    def test_forward_against_each_other(self):
        spatial = frechet_basis(
            coords,
            10,
        )
        coeffs = ref_spline(times)

        obs_direct = forward_obs(coeffs, spatial)

        temporal = BSpline.design_matrix(
            times,
            knots,
            3,
        ).T
        temporal = temporal.toarray()

        obs_base = forward_obs_time(
            ref_coeffs,
            spatial,
            temporal,
        )

        self.assertTrue(
            np.allclose(
                obs_direct,
                obs_base,
            )
        )


if __name__ == '__main__':
    unittest.main()
