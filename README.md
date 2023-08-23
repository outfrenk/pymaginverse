# geomagnetic_field_inversions
Library for the numerical inversion of geomagnetic field data. This code is based on Fortran Code provided by Monica Korte and two papers:
- Korte, M., & Constable, C. (2003). [Continuous global geomagnetic field models for the past 3000 years.](https://www.sciencedirect.com/science/article/pii/S0031920103001651) Physics of the Earth and Planetary Interiors, 140(1-3), 73-89. [10.1016/j.pepi.2003.07.013](https://doi.org/10.1016/j.pepi.2003.07.013)
- Korte, M., Donadini, F., & Constable, C. G. (2009). [Geomagnetic field for 0–3 ka: 2. A new series of time‐varying global models.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008GC002297) Geochemistry, Geophysics, Geosystems, 10(6). [10.1029/2008GC002297](https://doi.org/10.1029/2008GC002297)

## Library
The library consists of three main modules:
- `geomagnetic_field_inversions/data_prep.py`: contains methods for correctly preparing data for the `FieldInversion`-class. This method stores all required parameters per station into one class, which can then be imported directly into `FieldInversion`. The class requires the location of the station, together with either x, y, z, h-component magnetic data or declination, inclination, or intensity data per timestep. 
- `geomagnetic_field_inversions/field_inversion.py`: contains the `FieldInversion`-class that performs the actual inversion of geomagnetic field data. It requires, besides the time vector over which the inversion will take place, instances of `StationData` as minimum input.
- `geomagnetic_field_inversions/plot_tools.py`: contains basic plot tools for visualising data produced by the `FieldInversion`-class.
The `plot_tools.py` method allows easy plotting of:
- residuals
- Gauss coefficients through time
- spatial and temporal damping norms through time
- powerspectra and secular variation
- residual vs model norm plots (for choosing damping parameters)
- forward geomagnetic field maps of the world
- forward modeled data at specific locations

## Installation
You can install the library in two ways:
1. Cloning the repository and then `pip`-installing:
```
git clone https://github.com/geomagnetic_field_inversions
cd geomagnetic_field_inversions
pip install . -U
```

2. Cloning the repository and installing through poetry:
```
git clone https://github.com/geomagnetic_field_inversions
cd geomagnetic_field_inversions
poetry install
```
The package is *only* installed in a virtual environment created by poetry. Be sure to enter this environment by typing `poetry shell` into the terminal. In this environment you can start a jupyter notebook (type: `jupyter notebook`) and follow the tutorial: 

## Tutorial
We have provided a tutorial to make the library easier to use and understand. You can find the jupyter notebook containing the tutorial under `doc/Tutorial Geomagnetic Field Inversions.ipynb`.
This tutorial explains almost all methods of the three main modules and explains how to run a geomagnetic model with the `FieldInversion`-class.
