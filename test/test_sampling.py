from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from pandas import read_csv

from geomagnetic_field_inversions.data_prep import InputData
from geomagnetic_field_inversions import FieldInversion

from pymagglobal.utils import i2lm_l, i2lm_m


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

prior_samps = fInv.sample_prior(
    spat_damp=lambda_s,
    temp_damp=lambda_t,
)
posterior_samps = fInv.sample_posterior(
    spat_damp=lambda_s,
    temp_damp=lambda_t,
)

ind = 3

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle('Posterior')

fig.subplots_adjust(hspace=0)

axs[0].plot(
    fInv.knots[2:-2],
    prior_samps[:, ind, ::5],
    alpha=0.1,
    color='C0',
)
axs[0].plot(
    fInv.knots[2:-2],
    prior_samps[:, ind, 0],
    color='C0',
    label='Prior',
)

axs[1].plot(
    fInv.knots[2:-2],
    posterior_samps[:, ind, ::5],
    alpha=0.1,
    color='C0',
)
axs[1].plot(
    fInv.knots[2:-2],
    posterior_samps[:, ind].mean(axis=1),
    color='C0',
    label='Posterior',
)

for ax in axs:
    ax.set_ylabel(f'$g_{i2lm_l(ind):d}^{i2lm_m(ind):d}$ [nT]')
    ax.legend(frameon=False, loc='upper center')

plt.show()
