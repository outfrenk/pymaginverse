from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from pandas import read_csv

from pymaginverse.data_prep import InputData
from pymaginverse import FieldInversion

from pymagglobal import local_curve


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

iData = InputData(raw_data)
iData.compile_data()

lambda_s = 1.0e-13
lambda_t = 1.0e-3

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

myModel = fInv.result_to_pymagglobal('test')

# Honululu as lat, lon tuple
loc = (21.306944, -157.858333)

# Create an evenly spaced array over the ggfmb interval

times = np.linspace(
    myModel.t_min,
    myModel.t_max,
    501,
)
d, i, f = local_curve(times, loc, myModel)


fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(
    times,
    i,
)
ax.invert_xaxis()
ax.set_xlabel('time [ka BP]')
ax.set_ylabel('inclination [deg.]')

plt.show()
