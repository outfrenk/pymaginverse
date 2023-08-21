import numpy as np
from scipy.interpolate import BSpline, interp1d
import scipy.sparse as scs
import pandas as pd
from typing import Union, Final
from pathlib import Path
from tqdm import tqdm

from .data_prep import StationData
from .forward_modules import frechet, fwtools, rejection
from .damping_modules import damping
from .tools import geod2geoc as g2g


class FieldInversion:
    """
    Calculates geomagnetic field coefficients based on inputted data and
    damping parameters using the approach of Korte et al. (????)
    """

    def __init__(self,
                 time_array: Union[list, np.ndarray],
                 maxdegree: int = 3,
                 r_model: float = 6371.2,
                 geodetic: bool = True,
                 verbose: bool = False
                 ) -> None:
        """
        Initializes the Field Inversion class
        
        Parameters
        ----------
        time_array
            Sets timearray for the inversion in yr. Should be ascending
        maxdegree
            maximum order for spherical harmonics model, default 3
        r_model
            where the magnetic field is modeled (km distance from core)
        geodetic
            boolean specifying whether to use a geodetic coordinate frame. If
            True, geodetic coordinate frame is used and recalculated into a
            geocentric one. Otherwise, a geocentric frame is used.
            Default is geodetic (True)
        verbose
            Verbosity flag, defaults to False
        """
        # basic parameters
        self._SPL_DEGREE: Final[int] = 3

        # input parameters
        self.t_array = np.sort(time_array)
        self.maxdegree = maxdegree
        self.r_model = r_model
        self.geodetic = geodetic
        self.verbose = verbose

        # derived properties
        self._bspline = BSpline.basis_element(np.arange(self._SPL_DEGREE+2),
                                              extrapolate=False)
        self.data_array = np.zeros((0, len(time_array)))
        self.error_array = np.zeros((0, len(time_array)))
        self.time_cover = np.zeros((0, len(time_array)))
        self.accept_matrix = np.empty(0)
        self.types = []
        self.sc = 0  # station count
        self.types_ready = False
        self.types_sorted = np.empty(0)
        self.station_coord = np.zeros((0, 3))
        self.gcgd_conv = np.zeros((0, 2))
        self.spat_damp_matrix = np.empty(0)
        self.temp_damp_matrix = np.empty(0)
        self.spat_fac = np.empty(0)  # contains damping factors
        self.temp_fac = np.empty(0)
        self.spat_norm = np.empty(0)
        self.temp_norm = np.empty(0)
        self.spat_ddt = 0
        self.temp_ddt = 0
        self.splined_gh = np.empty(0)
        self.station_frechet = np.empty(0)
        self.res_iter = np.empty(0)
        self.unsplined_iter_gh = []
        self.dcname = []  # contains name of stations

    @property
    def maxdegree(self):
        return self._maxdegree

    @maxdegree.setter
    def maxdegree(self, degree: int):
        # determines the maximum number of spherical coefficients
        self._nm_total = int((degree+1)**2 - 1)
        self._maxdegree = int(degree)
        self.matrix_ready = False

    @property
    def t_array(self):
        return self._t_array

    @t_array.setter
    def t_array(self, array: Union[list, np.ndarray]):
        # check time array
        if len(array) == 1:
            raise Exception('t_array should consist or more than one timestep')
        self._t_step = array[1] - array[0]
        self._t_array = array
        # number of temporal splines
        self.nr_splines = len(array) + self._SPL_DEGREE - 1
        # location of timeknots
        self.time_knots = np.linspace(
            array[0] - self._SPL_DEGREE * self._t_step,
            array[-1] + self._SPL_DEGREE * self._t_step,
            num=len(array) + 2*self._SPL_DEGREE)
        # check for equally spaced time array
        for i in range(len(array)-1):
            if self._t_array[i+1] - self._t_array[i] != self._t_step:
                raise Exception("Time vector has different timesteps."
                                " Redefine vector with same timestep")
        self.times = len(array)
        self.matrix_ready = False

    def add_data(self,
                 data_class: StationData,
                 error_interp: str = 'linear'
                 ) -> None:
        """
        Adds data generated by the Station_data class

        Parameters
        ----------
        data_class
            instance of the Station_data class. Only added if it matches the
            time_array set in __init__
        error_interp
            string specifying interpolation of inputted error to time_array
            according to documentation of scipy.interpolate.interp1d

        Creates or modifies
        -------------------
        self.data_array
            contains the measurements per site
            size= (# datatypes, len(time vector)) (floats)
        self.error_array
            contains the error in measurements per site
            size= (# datatypes, len(time vector)) (float)
        self.time_cover
            contains whether the measurements per site cover the time array
            size= (# datatypes, len(time vector)) (boolean)
        self.types
            contains the type of all data in one long list
            size= # datatypes (integers)
        self.station_coord
            contains the colatitude, longitude, and radius of station
            size= (# datatypes, 3) (floats)
        self.gcgd_conv
            contains conversion factors for geodetic to geocentric conversion
            of magnetic components mx/dx and mz/dz
            size= (# datatypes, 2) (floats)
        self.types_ready
            boolean indicating if datatypes (self.types) are logically sorted
        """
        # translation datatypes
        typedict = {"x": 0, "y": 1, "z": 2, "hor": 3,
                    "int": 4, "inc": 5, "dec": 6}
        if isinstance(data_class, StationData):
            self.sc += 1
            # set up matrices
            data_entry = np.zeros((len(data_class.types), self.times))
            error_entry = np.ones((len(data_class.types), self.times))
            # time_cover indicates timerange of data (used in Fréchet)
            time_cover = np.zeros((len(data_class.types), self.times))
            types_entry = []
            name = data_class.__name__
            for c, types in enumerate(data_class.types):
                arg_min, arg_max = 0, self.times
                # check coverage data timevector begin
                if data_class.data[c][0][0] > self._t_array[0]:
                    if self.verbose:
                        print(f'{types} of {name} not covering start time')
                    if data_class.data[c][0][0] > self._t_array[-1]:
                        raise Exception(
                            f'{types} of {name} does not cover'
                            ' any timestep of timevector')
                    else:
                        # execute if not complete coverage
                        # _t_array elements should fall within min time
                        arg_min = np.min(np.argwhere((data_class.data[c][0][0]
                                                      - self._t_array) < 0))
                # check coverage data timevector end
                if data_class.data[c][0][-1] < self._t_array[-1]:
                    if self.verbose:
                        print(f'{types} of {name} not covering end time')
                    if data_class.data[c][0][-1] < self._t_array[0]:
                        raise Exception(
                            f'{types} of {name} does not cover'
                            ' any timestep of timevector')
                    else:
                        # execute if not complete coverage
                        # _t_array elements should fall within max time
                        arg_max = np.max(np.argwhere(data_class.data[c][0][-1]
                                                     - self._t_array) > 0)
                time_cover[c, arg_min:arg_max] = 1

                # Extract data from StationData-class
                if self.verbose:
                    print(f'Adding {types}-type')
                if types == 'inc':
                    # get inclination data at specified points in time
                    temp = data_class.fit_data[c](
                        self._t_array[arg_min:arg_max])
                    data_entry[c, arg_min:arg_max] = np.radians(temp)
                elif types == 'dec':
                    # get declination data at specified points in time
                    temp = data_class.fit_data[c](
                        self._t_array[arg_min:arg_max])
                    data_entry[c, arg_min:arg_max] = np.radians(temp)
                else:
                    data_entry[c, arg_min:arg_max] = data_class.fit_data[c](
                        self._t_array[arg_min:arg_max])
                # sample errors for time_array
                f = interp1d(data_class.data[c][0],
                             data_class.data[c][2],
                             kind=error_interp)
                error_entry[c, arg_min:arg_max] = f(
                    self._t_array[arg_min:arg_max])
                # count occurrence datatype and add to list
                types_entry.append(typedict[types])

            # change coordinates from geodetic to geocentric if required
            if self.geodetic:
                if self.verbose:
                    print(f'Coordinates are geodetic,'
                          ' translating to geocentric coordinates.')
                lat_geoc, r_geoc, cd, sd = g2g.latrad_in_geoc(
                    np.radians(data_class.lat), data_class.height)
                station_entry = np.array([0.5*np.pi - lat_geoc,
                                          np.radians(data_class.lon),
                                          r_geoc])
            else:
                if self.verbose:
                    print(f'Coordinates are geocentric,'
                          ' no translation required.')
                cd = 1.  # this will not change dx and dz when forming frechet
                sd = 0.
                station_entry = np.array([np.radians(90-data_class.lat),
                                          np.radians(data_class.lon),
                                          6371.2])

            # add data to attributes of the class if all is fine
            if self.verbose:
                print(f'Data of {name} is added to class')
            self.dcname.append(name)
            self.data_array = np.vstack((self.data_array, data_entry))
            self.error_array = np.vstack((self.error_array, error_entry))
            self.time_cover = np.vstack((self.time_cover, time_cover))
            self.types.append(types_entry)  # is now one long list
            self.station_coord = np.vstack((self.station_coord, station_entry))
            self.gcgd_conv = np.vstack((self.gcgd_conv, np.array([cd, sd])))
            self.types_ready = False
        else:
            raise Exception('data_class is not an instance of Station_Data')

    def prepare_inversion(self,
                          spat_dict: dict = None,
                          temp_dict: dict = None,
                          ) -> None:
        """
        Function to prepare matrices for the inversion

        Parameters
        ----------
        spat_dict
            dictionary for spatial damping containing the following keywords:
                df
                    damping factor to be applied to the total damping matrix
                damp_type
                    damping type to be applied
                ddt
                    derivative of B-Splines to be applied
                damp_dipole
                    boolean indicating whether to damp dipole coefficients.
        temp_dict
            dictionary for temporal damping containing the same keywords as
            spat_dict

        Creates or modifies
        -------------------
        self.spatddt, self.tempddt
            indicates requested derivative of the cubic B-Spline for damping
            (integer)
        self.station_frechet
            contains frechet matrix per location
            size= ((# stations x 3), nm_total) (floats)
        self.spat_fac, self.temp_fac
            contains the damping elements dependent on degree
             size= nm_total (floats) (see damp_types.py)
        self.spat_damp_matrix
            contains symmetric spatial damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.temp_damp_matrix
            contains symmetric temporal damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.matrix_ready
            indicates whether all matrices have been formed (boolean)
        self.types_ready
            boolean indicating if datatypes (self.types) are logically sorted
        """
        # check data and model space
        if spat_dict is None:
            spat_dict = {"df": 0, "damp_type": 'Gubbins',
                         "ddt": 0, "damp_dipole": False}
        if temp_dict is None:
            temp_dict = {"df": 0, "damp_type": 'Br2cmb',
                         "ddt": 2, "damp_dipole": True}
        self.spat_ddt, self.temp_ddt = spat_dict['ddt'], temp_dict['ddt']
        assert self._nm_total <= len(self.data_array), \
            'The spherical order of the model is too high,' \
            f' decrease maxdegree from {self._maxdegree} to a lower value.'

        # order datatypes in a more straightforward way
        if not self.types_ready:
            self.types_sorted = []
            for nr, stat in enumerate(self.types):
                for datum in stat:  # datum is 0 to 6
                    self.types_sorted.append(7*nr + datum)
            self.types_sorted = np.array(self.types_sorted)
            self.types_ready = True

        # calculate frechet dx, dy, dz for all stations
        if self.verbose:
            print('Calculating Schmidt polynomials and Fréchet coefficients')
        self.station_frechet = frechet.frechet_basis(
            self.station_coord, self._maxdegree)
        # geocentric correction
        dx, dz = g2g.frechet_in_geoc(
            self.station_frechet[:self.sc],
            self.station_frechet[2*self.sc:],
            self.gcgd_conv[:, 0], self.gcgd_conv[:, 1])
        self.station_frechet[:self.sc] = dx
        self.station_frechet[2*self.sc:] = dz

        # Prepare damping matrices
        if self.verbose:
            print('Calculating spatial damping matrix')
        if spat_dict['df'] != 0 and self._t_step != 0:
            self.spat_damp_matrix, self.spat_fac = damping.damp_matrix(
                self._maxdegree, self.nr_splines, self._t_step,
                spat_dict['df'], spat_dict['damp_type'], spat_dict['ddt'],
                damp_dipole=spat_dict['damp_dipole'])
            self.spat_damp_matrix = scs.csr_matrix(self.spat_damp_matrix)
        if self.verbose:
            print('Calculating temporal damping matrix')
        if temp_dict['df'] != 0 and self._t_step != 0:
            self.temp_damp_matrix, self.temp_fac = damping.damp_matrix(
                self._maxdegree, self.nr_splines, self._t_step,
                temp_dict['df'], temp_dict['damp_type'], temp_dict['ddt'],
                damp_dipole=temp_dict['damp_dipole'])
            self.temp_damp_matrix = scs.csr_matrix(self.temp_damp_matrix)
        self.matrix_ready = True
        if self.verbose:
            print('Calculations finished')

    def run_inversion(self,
                      x0: Union[np.ndarray, list],
                      max_iter: int = 10,
                      rej_crits: np.ndarray = None
                      ) -> None:
        """
        Runs the iterative inversion

        Parameters
        ----------
        x0
            starting model gaussian coefficients, should be a float or
            as long as (spherical_order + 1)^2 - 1
        max_iter
            maximum amount of iterations
        rej_crits
            Optional rejection criteria. Should be a length 7 array containing
            rejection criteria for x, y, z, hor, int, incl, and decl
            components. Criteria can be made time dependent by providing
            rejection criteria for every timestep. In that case shape should
            be (7, len(time vector))).

        Creates or modifies
        -------------------
        self.res_iter
             contains the RMS per datatype and the sum of all types
             size= 8 (floats)
        self.unsplined_iter_gh
            contains the BSpline function to unspline Gauss coeffs at any
            requested time (within range) for every iteration
            size= # iterations (BSpline functions)
        self.splined_gh
            contains the splined Gauss coeffs at all times of current iteration
            size= (len(nr_splines), nm_total) (floats)
        """
        # TODO: add uncertainty and data rejection
        if not self.matrix_ready:
            raise Exception('Matrices have not been prepared. '
                            'Please run prepare_inversion first.')

        self.res_iter = np.zeros((max_iter+1, 8))
        # initiate splined values with starting model
        if self.verbose:
            print('Setting up starting model')
        assert len(x0) == self._nm_total, \
            f'x0 has incorrect shape: {len(x0)},'\
            f' it should have length {self._nm_total}'
        self.splined_gh = np.zeros((self.nr_splines, self._nm_total))
        self.splined_gh[:] = x0
        self.accept_matrix = np.ones((len(self.types_sorted), self.times))
        for it in range(max_iter):
            if self.verbose:
                print(f'Start iteration {it+1}')
            rhs_matrix = np.zeros((self.times, self._nm_total))
            normal_eq_splined = np.zeros((self._nm_total * self.nr_splines,
                                          self._nm_total * self.nr_splines))

            rhs_spat_damp = -1*self.spat_damp_matrix*self.splined_gh.flatten()
            rhs_temp_damp = -1*self.temp_damp_matrix*self.splined_gh.flatten()
            gh_tstep = BSpline(c=self.splined_gh, t=self.time_knots,
                               k=self._SPL_DEGREE, axis=0,
                               extrapolate=False)(self._t_array)

            # Calculate frechet and residual matrix for all times
            # and apply time constraint
            if self.verbose:
                print('Create forward and residual observations')
            forwobs_matrix = fwtools.forward_obs(
                gh_tstep, self.station_frechet, reshape=False)
            frech_matrix = frechet.frechet_types(
                self.station_frechet, self.types_sorted, forwobs_matrix)
            forwobs_matrixrs = forwobs_matrix.reshape(
                7, self.sc, self.times).swapaxes(0, 1).reshape(
                7*self.sc, self.times)[self.types_sorted]
            res_matrix = fwtools.residual_obs(
                forwobs_matrixrs, self.data_array, self.types_sorted)
            res_matrix *= self.time_cover

            # reject data
            use_data_boolean = self.time_cover.copy()
            if rej_crits is not None:
                if self.verbose:
                    print('Apply rejection criteria')
                self.accept_matrix = rejection.reject_data(
                    res_matrix, self.types_sorted, rej_crits)
                use_data_boolean *= self.accept_matrix

            # apply rejection and time constraint to matrices
            frech_matrix *= np.repeat(use_data_boolean, self._nm_total, axis=1)
            res_matrix *= use_data_boolean
            res_weight = res_matrix / self.error_array
            # sum residuals
            count_type = np.zeros(7)
            type06 = self.types_sorted % 7
            for i in range(7):
                count_type[i] = np.sum(
                    np.where(type06 == i, use_data_boolean.T, 0))
            self.res_iter[it] = fwtools.residual_type(
                res_weight, self.types_sorted, count_type)

            # create time dependent matrices
            if self.verbose:
                print('Start formation time dependent matrices')
            for t in range(self.times):
                frech = frech_matrix[:, self._nm_total*t:self._nm_total*(t+1)]
                rhs_matrix[t] = np.matmul(
                    frech.T / self.error_array[:, t], res_weight[:, t])
                # Apply B-Splines straight away (much easier)
                for j in range(self._SPL_DEGREE):
                    for k in range(self._SPL_DEGREE):
                        normal_eq_splined[
                            (t+j) * self._nm_total:(t+j+1) * self._nm_total,
                            (t+k) * self._nm_total:(t+k+1) * self._nm_total
                        ] += np.matmul(frech.T*self._bspline(j+1)
                                       / self.error_array[:, t]**2,
                                       frech*self._bspline(k+1))
            normal_eq_splined = scs.csr_matrix(normal_eq_splined)
            rhs_splined = np.zeros(self.nr_splines * self._nm_total)
            for i in range(self._nm_total):
                rhs_splined[i::self._nm_total] = np.convolve(
                    rhs_matrix[:, i], self._bspline(np.arange(
                        1, self._SPL_DEGREE+1)))
            # add spatial and temporal damping to the matrix and vector
            normal_eq_splined += self.spat_damp_matrix + self.temp_damp_matrix
            rhs_splined += rhs_spat_damp + rhs_temp_damp

            # solve the equations
            if self.verbose:
                print('Solve equations')
            update = scs.linalg.spsolve(normal_eq_splined, rhs_splined)
            self.splined_gh = (self.splined_gh.flatten() + update).reshape(
                self.nr_splines, self._nm_total)
            # cut of the sides that do not have physical meaning
            spline = BSpline(t=self.time_knots, c=self.splined_gh,
                             k=3, axis=0, extrapolate=False)
            self.unsplined_iter_gh.append(spline)

            if self.verbose:
                print('Residual is %.2f' % self.res_iter[it, 7])
            # residual after last iteration
            if it == max_iter - 1:
                if self.verbose:
                    print('Calculate residual last iteration')
                gh_tstep = BSpline(c=self.splined_gh, t=self.time_knots,
                                   k=self._SPL_DEGREE, axis=0,
                                   extrapolate=False)(self._t_array)
                forwobs_matrixrs = fwtools.forward_obs(
                    gh_tstep, self.station_frechet, reshape=True
                )[self.types_sorted]
                res_matrix = fwtools.residual_obs(
                    forwobs_matrixrs, self.data_array, self.types_sorted)
                res_matrix *= use_data_boolean
                res_weight = res_matrix / self.error_array
                # sum residuals
                self.res_iter[it+1] = fwtools.residual_type(
                    res_weight, self.types_sorted, count_type)
                if self.verbose:
                    print('Residual is %.2f' % self.res_iter[it+1, 7])
                    print('Calculating spatial and temporal norms')
                self.spat_norm = damping.damp_norm(
                    self.spat_fac, self.splined_gh, self.spat_ddt, self._t_step
                )
                self.temp_norm = damping.damp_norm(
                    self.temp_fac, self.splined_gh, self.temp_ddt, self._t_step
                )
                if self.verbose:
                    print('Finished inversion')

    def save_coefficients(self,
                          basedir: Union[Path, str] = '.',
                          file_name: str = 'coeff',
                          save_iterations: bool = True,
                          save_residual: bool = False,
                          rejection_report: bool = False,
                          ) -> None:
        """
        Save the Gauss coefficients at every timestep

        Parameters
        ----------
        basedir
            path where files will be saved
        file_name
            optional name to add to files
        save_iterations
            boolean indicating whether to save coefficients after
            each iteration. Is saved with the following shape:
             (# iterations, len(time vector), nm_total)
        save_residual
            boolean indicating whether to save the residuals of each timestep
        rejection_report
            boolean indicating whether to store a rejection report showing
            which data has been rejected.
        """
        dict_types = ['x', 'y', 'z', 'hor', 'int', 'incl', 'decl']
        # save residual
        if save_residual:
            residual_frame = pd.DataFrame(
                self.res_iter, columns=['res x', 'res y', 'res z', 'res hor',
                                        'res int', 'res incl', 'res decl',
                                        'res total'])
            residual_frame.to_csv(basedir / f'{file_name}_residual.csv',
                                  sep=';')
        if rejection_report:
            f = open(basedir / f'{file_name}_reject.txt', 'w')
            row = 0
            for n, name in enumerate(self.dcname):
                f.write(f'Station {name} \n')
                for types in self.types[n]:
                    datarow = self.accept_matrix[row]
                    if np.sum(datarow) != len(datarow):
                        rejecttime = np.where(self.accept_matrix[row] == 0)[0]
                        f.write(f'{dict_types[types]}:'
                                f' {self._t_array[rejecttime]} \n')
                    row += 1
            f.close()
        if save_iterations:
            all_coeff = np.zeros((
                len(self.unsplined_iter_gh), self.times, self._nm_total))
            for i in range(len(self.unsplined_iter_gh)):
                all_coeff[i] = self.unsplined_iter_gh[i](self._t_array)
            np.save(basedir / f'{file_name}_all.npy', all_coeff)
        else:
            gh_time = self.unsplined_iter_gh[-1](self._t_array)
            np.save(basedir / f'{file_name}_final.npy', gh_time)

    def sweep_damping(self,
                      x0: Union[list, np.ndarray],
                      spatial_range: Union[list, np.ndarray],
                      temporal_range: Union[list, np.ndarray],
                      spat_dict: dict = None,
                      temp_dict: dict = None,
                      max_iter: int = 5,
                      basedir: Union[str, Path] = '.',
                      overwrite: bool = False
                      ) -> None:
        """ Sweep through damping parameters to find ideal set

        Parameters
        ----------
        x0
            starting model gaussian coefficients, should be a float or
            as long as (spherical_order + 1)^2 - 1
        spatial_range
            array or list to vary spatial damping parameters. Can be None if
            temporal_range is inputted
        temporal_range
            array or list to vary temporal damping parameters.  Can be None if
            spatial_range is inputted
        spat_dict, temp_dict
            dictionary for spatial, temporal damping
            see prepare_inversion for more info
        max_iter
            maximum number of iterations. defaults to 5 iterations
        basedir
            path where files will be saved
        overwrite
            boolean indicating whether to overwrite existing files with
            exactly the same damping parameters. otherwise set of damping
            parameters is skipped over in the calculations.
        """
        if spat_dict is None:
            spat_dict = {"damp_type": 'Gubbins', "ddt": 0,
                         "damp_dipole": False}
        if temp_dict is None:
            temp_dict = {"damp_type": 'Br2cmb', "ddt": 2, "damp_dipole": True}

        for spatial_df in tqdm(spatial_range):
            spat_dict['df'] = spatial_df
            for temporal_df in temporal_range:
                temp_dict['df'] = temporal_df
                if overwrite or not (basedir / f'{spatial_df:.2e}s+'
                                               f'{temporal_df:.2e}t_final.npy'
                                     ).is_file():
                    self.prepare_inversion(spat_dict, temp_dict)
                    self.run_inversion(x0, max_iter)
                    self.save_coefficients(
                        file_name=f'{spatial_df:.2e}s+{temporal_df:.2e}t',
                        basedir=basedir, save_iterations=False,
                        save_residual=True)
