# Installation guideline

In order to get the inversion code running, you need to have the Clang compiler and the OpenMP libraries installed. 
The easiest way to get a working set up is using the conda dependency manager, e.g. from [here](https://github.com/conda-forge/miniforge).
Replacing <env-name> with a name of your choice, you can create a conda environment with all dependencies provided using:  
```
conda create --name <env-name> clang cython pyshtools llvm-openmp "python<=3.11"
```
Then you can run
```
pip install -e .
```
in the root directory of the repository, to install the package.
