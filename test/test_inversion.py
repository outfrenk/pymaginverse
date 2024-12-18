import unittest
from pathlib import Path

import numpy as np
from pandas import read_csv

from pymaginverse.data_prep import InputData
from pymaginverse import FieldInversion

path = Path(__file__).parent

with open(path / '../benchmark_data/iteration_1', 'r') as fh:
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

with open(path / '../benchmark_data/iteration_10', 'r') as fh:
    fh.readline()
    # read the rest as a long string
    input_string = ' '.join(fh.readlines())
    # create an array from the string
    input_array = np.fromstring(input_string, dtype=float, sep=' ')

    ref_coeffs_10 = input_array[n_spl+4+2:n_spl*(n_coeffs+1)+4+2]
    ref_coeffs_10 = ref_coeffs_10.reshape(n_spl, n_coeffs)

raw_data = read_csv(path / '../benchmark_data/testdata.csv')
raw_data = raw_data.query(f'{t_min} <= t <= {t_max}')
raw_data.reset_index(inplace=True, drop=True)

raw_data.loc[:, 'D'] /= 10
raw_data.loc[:, 'dD'] /= 10
raw_data.loc[:, 'I'] /= 10
raw_data.loc[:, 'dI'] /= 10

iData = InputData(raw_data)
iData.compile_data()


class Test_inversion(unittest.TestCase):
    def test_single_inversion(self):
        lambda_s = 1.0e-13
        lambda_t = 1.0e-3 / 4 / np.pi

        fInv = FieldInversion(
            t_min=t_min, t_max=t_max, t_step=knots[1]-knots[0],
            maxdegree=10,
            verbose=False,
        )

        fInv.prepare_inversion(
            iData,
            spat_type="ohmic_heating",
            temp_type="min_acc",
        )

        x0 = np.zeros(fInv._nr_coeffs)
        x0[0] = -30e3
        fInv.run_inversion(
            x0,
            max_iter=1,
            spat_damp=lambda_s,
            temp_damp=lambda_t,
        )

        res_coeffs = fInv.coeffs_solution

        self.assertTrue(
            np.allclose(
                ref_coeffs,
                res_coeffs,
                atol=25,
            )
        )

    def test_ten_inversions(self):
        lambda_s = 1.0e-13
        lambda_t = 1.0e-3 / 4 / np.pi

        fInv = FieldInversion(
            t_min=t_min, t_max=t_max, t_step=knots[1]-knots[0],
            maxdegree=10,
            verbose=False,
        )

        fInv.prepare_inversion(
            iData,
            spat_type="ohmic_heating",
            temp_type="min_acc",
        )

        x0 = np.zeros(fInv._nr_coeffs)
        x0[0] = -30e3
        fInv.run_inversion(
            x0,
            max_iter=10,
            spat_damp=lambda_s,
            temp_damp=lambda_t,
        )

        res_coeffs = fInv.coeffs_solution

        self.assertTrue(
            np.allclose(
                ref_coeffs_10,
                res_coeffs,
                atol=25,
            )
        )


if __name__ == '__main__':
    unittest.main()
