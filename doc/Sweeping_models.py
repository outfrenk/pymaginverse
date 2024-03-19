import numpy as np
import pandas as pd
from pathlib import Path
from geomagnetic_field_inversions import InputData, FieldInversion

path = Path().absolute()
dataset = pd.read_csv(path / 'example_data.csv', index_col=0)
inputdata = InputData(dataset)

sweep_inv = FieldInversion(t_min=-2000, t_max=1990, t_step=10, maxdegree=10)
sweep_inv.prepare_inversion(inputdata, spat_type='ohmic_heating',
                            temp_type='min_acc')
# set up our range of damping parameters
spatial_range = np.logspace(-17, -11, 7)
temporal_range = np.logspace(-7, 1, 9)
# set starting model
x0 = np.zeros(sweep_inv._nm_total)
x0[0] = -30000
# call sweep_damping, results will be saved in the path / 'output'-folder
sweep_inv.sweep_damping(x0, spatial_range, temporal_range,
                        basedir= path / 'output')