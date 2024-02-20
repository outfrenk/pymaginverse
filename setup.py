# conda install
# openmp pyshtools Cython clang
# llvm-openmp
# conda create --name gmc "python<=3.11" openmp pyshtools Cython clang
# llvm-openmp

import setuptools
# from setuptools.extension import Extension
import os
from Cython.Build import cythonize
import numpy as np

os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'
os.environ['LDSHARED'] = 'clang -shared'

ext_modules = [
    setuptools.Extension(
        "geomagnetic_field_inversions.banded_tools.build_banded",
        ["geomagnetic_field_inversions/banded_tools/build_banded.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-mavx', '-fopenmp', '-ffast-math'],
        extra_link_args=['-fopenmp'],
    )
]


with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='geomagnetic_field_inversions',
    version='1.0.0',
    description=('Python lib to perform geomagnetic field inversions'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='F. Out, L. B. van Grinsven, M. Korte, L. V. de Groot',
    author_email='f.out@uu.nl',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'pathlib',
        'tqdm',
        'pyshtools',
        'cartopy',
        'numba',
        'openpyxl',
        'pytest'
    ],
    # TODO: Update license
    classifiers=[
        'License :: BSD2 License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"},
    ),
)
