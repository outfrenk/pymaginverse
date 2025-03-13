[![DOI](https://zenodo.org/badge/612286043.svg)](https://zenodo.org/doi/10.5281/zenodo.11098494)
[![PyPI](https://img.shields.io/pypi/v/pymaginverse.svg)](https://pypi.org/project/pymaginverse/)
# pymaginverse
Library for the numerical inversion of geomagnetic field data. This library is brough to you by Frenk Out, Maximilian Schanner, Liz van Grinsven, Monika Korte, and Lennart de Groot. This code is based on Fortran code used for the following two papers:
- Korte, M., & Constable, C. (2003). [Continuous global geomagnetic field models for the past 3000 years.](https://www.sciencedirect.com/science/article/pii/S0031920103001651) Physics of the Earth and Planetary Interiors, 140(1-3), 73-89. [10.1016/j.pepi.2003.07.013](https://doi.org/10.1016/j.pepi.2003.07.013)
- Korte, M., Donadini, F., & Constable, C. G. (2009). [Geomagnetic field for 0–3 ka: 2. A new series of time‐varying global models.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008GC002297) Geochemistry, Geophysics, Geosystems, 10(6). [10.1029/2008GC002297](https://doi.org/10.1029/2008GC002297)

## Library
The library consists of two main modules:
- `pymaginverse/data_prep.py`: contains methods for correctly preparing data for the `FieldInversion`-class. This method stores all required parameters per station into one class, which can then be imported directly into `FieldInversion`. The class takes a csv-file of declination, inclination, intensity, x, y, z, and/or h-component magnetic data + accompanying data errors.
- `pymaginverse/field_inversion.py`: contains the `FieldInversion`-class that performs the actual inversion of geomagnetic field data. It requires, besides the time vector over which the inversion will take place, an instance of `InputData` as minimum input.

## Installation
### option a: pip
The easiest way to get a working environment set up is using the conda dependency manager, e.g. from [here](https://github.com/conda-forge/miniforge).
Replacing <env-name> with a name of your choice, you can create a conda environment with all dependencies provided using:
```
conda create --name <env-name> "python==3.11"
```

Then, activate your environment
```
conda activate <env-name>
```

Subsequently, either enter the root directory of the repository (which you should have downloaded with either git clone or the download button above), to install the package:
```
pip install . -U
```

or pip install directly from the pypi repository:
```
pip install pymaginverse
```

### Option b: poetry
In order to get the inversion code running with poetry, you need to have poetry *properly* installed. See the [website](https://python-poetry.org/docs/#installing-with-the-official-installer) of poetry, or run:
```
curl -sSL https://install.python-poetry.org | python3 -
```

The easiest way to get a working environment set up is using the conda dependency manager, e.g. from [here](https://github.com/conda-forge/miniforge).
Replacing <env-name> with a name of your choice, you can create a conda environment with all dependencies provided using:
```
conda create --name <env-name> "python==3.11"
```

Then, activate your environment
```
conda activate <env-name>
```

Enter the root directory of the repository (which you should have downloaded with either git clone or the download button above), to install the package.
```
poetry install
```

## Uninstallation
Uninstallation is possible by typing in the terminal (possibly in the specific virtual environment):
```
pip uninstall pymaginverse
```
## Tutorial
We have provided four tutorials to make the library easier to use and understand. You can find the jupyter notebooks containing the tutorials in the `doc`-folder.
The tutorials are:
1. Tutorial 1: loading data and running a model. In this Tutorial we show how to load data into the `InputData`-class, and teach you the basics of the `FieldInversion`-class (including several damping types).
2. Tutorial 2: plotting results. We show some basic plotting tools allowing a better understanding of the created geomagnetic model. Examples consist of residual plots, gauss coefficients, powerspectra, and global magnetic field maps. 
3. Tutorial 3: sweeping through models. We show how to loop through models with different damping parameters to find a optimal model with the help of the elbow-plot and powerspectra.
4. Tutorial 4: loading geomagia dataset. We show how to load a geomagia dataset with our `InputData`-class.

All files required for the tutorials are also found in the `doc`-folder.

## Testing
All tests for this library can be found in the `test`-folder. To run the tests you need to have pytest installed.
After installation of pytest, you can test the code by typing `pytest`.

## Accompanying Article
The Article describing this library can be found here:
Out, F., Schanner, M., van Grinsven, L., Korte, M., & de Groot, L. V. (2025). Pymaginverse: A python package for global geomagnetic field modeling. Applied Computing and Geosciences, 100222.
doi: [10.1016/j.acags.2025.100222](https://doi.org/10.1016/j.acags.2025.100222)

## Acknowledgements
This Python code is based on a version of Fortran codes that have been spread within the geomagnetic community by personal communication and in its original version were mainly written by David Gubbins, Kathryn Whaler, Jeremy Bloxham, and Andrew Jackson. The authors want to express their gratitude to Sanja Panovska for the fruitful discussions on the algorithm and its spatial and temporal damping options.
