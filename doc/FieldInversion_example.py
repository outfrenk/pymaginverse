import numpy as np
import pandas as pd
from pathlib import Path
from geomagnetic_field_inversions import InputData, FieldInversion

# load csv-file into InputData class
path = Path().absolute()
dataset = pd.read_csv(path / 'example_data.csv', index_col=0)
inputdata = InputData(dataset)

##### start geomagnetic field inversion #####
# set time array and maximum spherical degree
test_inv = FieldInversion(t_min=-2000, t_max=1990, t_step=10, maxdegree=10)
# load data-class and set damping types: Ohmic heating and minimum acceleration
test_inv.prepare_inversion(inputdata, spat_type='ohmic_heating',
                           temp_type='min_acc')
# set starting model, should have 120 elements
x0 = np.zeros(test_inv._nr_coeffs)
x0[0] = -30000
# run inversion by setting start model, damp factors, and max # iterations
test_inv.run_inversion(x0, spat_damp=1.0e-13, temp_damp=1.0e-3, max_iter=5)
# save gauss coefficients results
test_inv.save_coefficients(path / 'output', file_name='example')