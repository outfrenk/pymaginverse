import setuptools
# from setuptools.extension import Extension
import sys

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
    install_requires=['matplotlib',
                      'numpy',
                      'scipy',
                      'pandas',
                      'pathlib',
                      'tqdm',
                      'pyshtools',
                      'cartopy',
					  'numba',
                      'openpyxl',
                      'pytest'],
    # TODO: Update license
    classifiers=['License :: BSD2 License',
                 'Programming Language :: Python :: 3 :: Only',
                 ],
)
