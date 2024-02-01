import unittest
from pathlib import Path

import numpy as np
from pandas import read_csv

from geomagnetic_field_inversions import StationData, FieldInversion

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

raw_data = read_csv(path / '../benchmark_data/testdata.csv')
raw_data = raw_data.query(f'{t_min} <= t <= {t_max}')
raw_data.reset_index(inplace=True, drop=True)

raw_data.loc[:, 'D'] /= 10
raw_data.loc[:, 'dD'] /= 10
raw_data.loc[:, 'I'] /= 10
raw_data.loc[:, 'dI'] /= 10


class Test_inversion(unittest.TestCase):
    def test_single_inversion(self):
        lambda_s = 1.0e-13
        lambda_t = 1.0e-3

        fInv = FieldInversion(
            time_array=knots[3:-3],
            maxdegree=10,
            verbose=False,
        )
        for it, row in raw_data.iterrows():
            _station = StationData(
                row['lat'],
                row['lon'],
            )
            if not np.isnan(row['D']):
                _station.add_data(
                    'dec',
                    [[row['t']], [row['D']]],
                    1,  # time_factor
                    error=[row['dD']],
                )
            if not np.isnan(row['I']):
                _station.add_data(
                    'inc',
                    [[row['t']], [row['I']]],
                    1,  # time_factor
                    error=[row['dI']],
                )
            if not np.isnan(row['F']):
                _station.add_data(
                    'int',
                    [[row['t']], [row['F']]],
                    1,  # time_factor
                    error=[row['dF']],
                )
            fInv.add_data(_station)

        fInv.prepare_inversion(
            spat_fac=lambda_s,
            temp_fac=lambda_t,
        )

        x0 = np.zeros(fInv._nm_total)
        x0[0] = -30000
        fInv.run_inversion(x0, max_iter=0)

        res_coeffs = fInv.unsplined_iter_gh[0].c

        self.assertTrue(
            np.allclose(
                ref_coeffs,
                res_coeffs,
                rtol=5e-3,
                atol=10,
            )
        )


if __name__ == '__main__':
    unittest.main()
