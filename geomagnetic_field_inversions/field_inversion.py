import numpy as np
import math
from scipy.integrate import newton_cotes
from scipy.interpolate import BSpline, interp1d
from scipy.linalg import pinv
import scipy.sparse as scs
import pandas as pd
from typing import Union, Final
from pathlib import Path
from tqdm import tqdm

from .data_prep import StationData
from .forward_modules import frechet
from .damping_modules import damping
from .geodtogeoc_modules import geod2geoc_conv as g2g


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
        self.types = []
        self.count_type = np.zeros(7)  # counts occurrence datatypes all times
        self.types_ready = False
        self.types_sorted = np.empty(0)
        self.station_coord = np.zeros((0, 3))
        self.gcgd_conv = np.zeros((0, 2))
        self.spat_damp_matrix = np.empty(0)
        self.temp_damp_matrix = np.empty(0)
        self.splined_gh = np.empty(0)
        self.unsplined_gh = np.empty(0)
        self.station_frechet = np.empty(0)
        self.res_iter = np.empty(0)
        self.unsplined_iter = np.empty(0)

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
            self._t_step = 0
            self._t_array = array
        else:
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
            for i in range(len(self._t_array)-1):
                if self._t_array[i+1] - self._t_array[i] != self._t_step:
                    raise Exception("Time vector has different timesteps."
                                    " Redefine vector with same timestep")
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
        self.count_type
            counts the occurrence of the 7 different datatypes
            size= 7 (integers)
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
            # set up matrices
            data_entry = np.zeros((len(data_class.types), len(self._t_array)))
            error_entry = np.ones((len(data_class.types), len(self._t_array)))
            # time_cover indicates timerange of data (used in Fréchet)
            time_cover = np.zeros((len(data_class.types), len(self._t_array)))
            types_entry = []
            for c, types in enumerate(data_class.types):
                arg_min, arg_max = 0, len(self._t_array)
                # check coverage data timevector begin
                if data_class.data[c][0][0] > self._t_array[0]:
                    if self.verbose:
                        print(f'{types} of {data_class.__name__} not covering'
                              ' start time')
                    if data_class.data[c][0][0] > self._t_array[-1]:
                        raise Exception(
                            f'{types} of {data_class.__name__} does not cover'
                            ' any timestep of timevector')
                    else:
                        # execute if not complete coverage
                        # _t_array elements should fall within min time
                        arg_min = np.min(np.argwhere((data_class.data[c][0][0]
                                                      - self._t_array) < 0))
                # check coverage data timevector end
                if data_class.data[c][0][-1] < self._t_array[-1]:
                    if self.verbose:
                        print(f'{types} of {data_class.__name__} not covering'
                              ' end time')
                    if data_class.data[c][0][-1] < self._t_array[0]:
                        raise Exception(
                            f'{types} of {data_class.__name__} does not cover'
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
                self.count_type[typedict[types]] += np.sum(time_cover[c])

            # change coordinates from geodetic to geocentric if required
            if self.geodetic:
                if self.verbose:
                    print(f'Coordinates are geodetic,'
                          ' translating to geocentric coordinates.')
                lat_geoc, r_geoc, cd, sd = g2g.latrad_in_geoc(
                    math.radians(data_class.lat), data_class.height)
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
                print(f'Data of {data_class.__name__} is added to class')
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
        self.station_frechet
            contains frechet matrix per location
            size= ((# stations x 3), nm_total) (floats)
        self.spat_damp_matrix
            contains symmetric spatial damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.temp_damp_matrix
            contains symmetric temporal damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.matrix_ready
            indicates whether all matrices have been formed (boolean)
        """
        # check data and model space
        if spat_dict is None:
            spat_dict = {"df": 0, "damp_type": 'Gubbins',
                         "ddt": 0, "damp_dipole": False}
        if temp_dict is None:
            temp_dict = {"df": 0, "damp_type": 'Br2cmb',
                         "ddt": 2, "damp_dipole": True}
        assert self._nm_total <= len(self.data_array), \
            'The spherical order of the model is too high,' \
            f' decrease maxdegree from {self._maxdegree} to a lower value.'

        # order datatypes in a more straigtforward way
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
        self.station_frechet = frechet.frechet_formation(
            self.station_coord, self._maxdegree)
        # geocentric correction
        dx, dz = g2g.frechet_in_geoc(
            self.station_frechet[:len(self.station_coord)],
            self.station_frechet[2*len(self.station_coord):],
            self.gcgd_conv[:, 0], self.gcgd_conv[:, 1])
        self.station_frechet[:len(self.station_coord)] = dx
        self.station_frechet[2*len(self.station_coord):] = dz

        # Prepare damping matrices
        if self.verbose:
            print('Calculating spatial damping matrix')
        if spat_dict['df'] != 0 and self._t_step != 0:
            self.spat_damp_matrix = damping.damp_matrix(
                self._maxdegree, self.nr_splines, self._t_step,
                spat_dict['df'], spat_dict['damp_type'], spat_dict['ddt'],
                damp_dipole=spat_dict['damp_dipole'])
            self.spat_damp_matrix = scs.csr_matrix(self.spat_damp_matrix)
        if self.verbose:
            print('Calculating temporal damping matrix')
        if temp_dict['df'] != 0 and self._t_step != 0:
            self.temp_damp_matrix = damping.damp_matrix(
                self._maxdegree, self.nr_splines, self._t_step,
                temp_dict['df'], temp_dict['damp_type'], temp_dict['ddt'],
                damp_dipole=temp_dict['damp_dipole'])
            self.temp_damp_matrix = scs.csr_matrix(self.temp_damp_matrix)
        self.matrix_ready = True
        if self.verbose:
            print('Calculations finished')

    def run_inversion(self,
                      x0: Union[np.ndarray, list],
                      max_iter: int = 5,
                      **prep_kwargs
                      ) -> None:
        """
        Run the iterative inversion

        Parameters
        ----------
         x0
            starting model gaussian coefficients, should be a float or
            as long as (spherical_order + 1)^2 - 1
        max_iter
            maximum amount of iterations
        **prep_kwargs
            optional keyword arguments in case the prepare_inversion function
            has not been run yet. See prepare_inversion for more information.

        """
        # TODO: add uncertainty and data rejection
        if self._t_step == 0:
            raise Exception('Switch to function "run_inversion_notime" to'
                  ' execute calculations for one timestep')
        if not self.matrix_ready:
            if self.verbose:
                print('Preparing matrices for iterative inversion')
            self.prepare_inversion(**prep_kwargs)

        self.res_iter = np.zeros((max_iter+1, 8))
        self.unsplined_iter = np.zeros(
            (max_iter, self._nm_total, len(self._t_array)))
        # initiate splined values with starting model
        if self.verbose:
            print('Setting up starting model')
        assert len(x0) == self._nm_total, \
            f'x0 has incorrect shape: {len(x0)},'\
            f' it should have length {self._nm_total}'
        self.splined_gh = np.zeros((self.nr_splines, self._nm_total))
        self.splined_gh[:] = x0
        for it in range(max_iter):
            if self.verbose:
                print(f'Start iteration {it+1}')
            rhs_matrix = np.zeros((len(self._t_array), self._nm_total))
            normal_eq_splined = np.zeros((self._nm_total * self.nr_splines,
                                          self._nm_total * self.nr_splines))

            rhs_spat_damp = -1*self.spat_damp_matrix*self.splined_gh.flatten()
            rhs_temp_damp = -1*self.temp_damp_matrix*self.splined_gh.flatten()
            gh_timesteps = BSpline(c=self.splined_gh, t=self.time_knots,
                                   k=self._SPL_DEGREE, axis=0,
                                   extrapolate=False)(self._t_array)
            print(gh_timesteps)
            # Calculate frechet and residual matrix for all times
            frech_matrix, res_matrix = frechet.forward_obs(
                self.data_array, gh_timesteps, self.station_frechet,
                self.types_sorted)
            # apply time constraint
            frech_matrix *= np.repeat(self.time_cover, self._nm_total, axis=1)
            res_matrix *= self.time_cover
            res_weight = res_matrix / self.error_array
            print('residual_weighted', res_matrix)
            # sum residuals
            type06 = self.types_sorted % 7
            for i in range(7):
                if self.count_type[i] != 0:
                    self.res_iter[it, i] = np.sqrt(np.sum(
                        res_weight[type06 == i]**2) / self.count_type[i])
            self.res_iter[it, 7] = np.sqrt(
                np.sum(res_weight ** 2) / sum(self.count_type))

            for t in range(len(self._t_array)):
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
            update = scs.linalg.spsolve(normal_eq_splined, rhs_splined)
            print('update:', update)
            self.splined_gh = (self.splined_gh.flatten() + update).reshape(
                self.nr_splines, self._nm_total)
            self.unsplined_gh = []
            # cut of the sides that do not have physical meaning
            for gh in range(self._nm_total):
                spline = BSpline(t=self.time_knots, c=self.splined_gh[:, gh],
                                 k=3, extrapolate=False)
                self.unsplined_iter[it, gh, :] = spline(self._t_array)
                self.unsplined_gh.append(spline)
            if self.verbose:
                print('Residual is %.2f' % self.res_iter[it, 7])
            # residual after last iteration
            if it == max_iter - 1:
                if self.verbose:
                    print('Calculate residual last iteration')
                gh_timesteps = BSpline(c=self.splined_gh, t=self.time_knots,
                                       k=self._SPL_DEGREE, axis=0,
                                       extrapolate=False)(self._t_array)
                frech_matrix, res_matrix = frechet.forward_obs(
                    self.data_array, gh_timesteps, self.station_frechet,
                    self.types_sorted)
                # apply time constraint
                res_matrix *= self.time_cover
                res_weight = res_matrix / self.error_array
                # sum residuals
                for i in range(7):
                    if self.count_type[i] != 0:
                        self.res_iter[it+1, i] = np.sqrt(np.sum(
                            res_weight[type06 == i]**2) / self.count_type[i])
                self.res_iter[it+1, 7] = np.sqrt(
                    np.sum(res_weight ** 2) / sum(self.count_type))
                if self.verbose:
                    print('Residual is %.2f' % self.res_iter[it+1, 7])

    def run_inversion_notime(self,
                             x0: Union[np.ndarray, list] = None,
                             max_iter: int = 5,
                             int_mult: float = 1):
        """
        Run the iterative inversion for a single time

        Parameters
        ----------
         x0
            starting model gaussian coefficients, should be a float or
            as long as (spherical_order + 1)^2 - 1
        max_iter
            maximum amount of iterations
        int_mult
            multiplies intensity values with this parameter, default 1

        """
        if self.matrix_ready is False:
            if self.verbose:
                print('Preparing matrices for iterative inversion')
            self.prepare_inversion(spatial_df=0, temporal_df=0)

        self.res_iter = np.zeros((max_iter + 1, 8))
        self.unsplined_iter = np.zeros((max_iter, self._nm_total))
        # initiate splined values with starting model
        assert len(x0) == self._nm_total, \
            f'x0 has incorrect shape: {len(x0)},' \
            f' it should be: {self._nm_total}'
        self.unsplined_gh = x0
        for it in range(max_iter):
            if self.verbose:
                print(f'Start iteration {it + 1}')
            count_all = np.zeros(7)

            # Calculate the forward observation
            forw_obs, frechet_matrix, res_obs, count = \
                self.forward_frechet(self.unsplined_gh, 0, it)
            count_all += count
            # save residual
            self.res_iter[it, 7] += np.sum(
                (res_obs / self.error_array[:, 0]) ** 2)
            # multiply the 'right hand side' and apply covariance matrix
            rhs_unsplined = np.matmul(
                frechet_matrix.T / self.error_array[:, 0],
                res_obs / self.error_array[:, 0])
            normal_eq_unsplined = np.matmul(frechet_matrix.T
                                            / self.error_array[:, 0]**2,
                                            frechet_matrix)
            for i in range(7):
                if count_all[i] != 0:
                    self.res_iter[it, i] = np.sqrt(
                        self.res_iter[it, i] / count_all[i])
            self.res_iter[it, 7] = np.sqrt(self.res_iter[it, 7]
                                                  / np.sum(count_all))

            # solve the equations
            update = np.matmul(pinv(normal_eq_unsplined), rhs_unsplined)
            self.unsplined_gh += update
            # cut of the sides that do not have physical meaning
            self.unsplined_iter[it, :] = self.unsplined_gh
            if self.verbose:
                print('Residual is %.2f' % self.res_iter[it, 7])
            # residual after last iteration
            if it == max_iter - 1:
                if self.verbose:
                    print('Calculate residual last iteration')
                count_all = np.zeros(7)
                forw_obs, frechet_matrix, res_obs, count = \
                    self.forward_frechet(self.unsplined_gh, 0, it + 1)
                count_all += count
                self.res_iter[it + 1, 7] += np.sum(
                    (res_obs / self.error_array[:, 0]) ** 2)
                for i in range(7):
                    if count_all[i] != 0:
                        self.res_iter[it + 1, i] = np.sqrt(
                            self.res_iter[it + 1, i] / count_all[i])
                self.res_iter[it + 1, 7] = \
                    np.sqrt(
                        self.res_iter[it + 1, 7] / np.sum(count_all))
                if self.verbose:
                    print('Residual is %.2f' % self.res_iter[it + 1, 7])

    def save_spherical_coefficients(self,
                                    basedir: Union[Path, str] = '.',
                                    file_name: str = 'result',
                                    save_iterations: bool = True,
                                    save_residual: bool = False):
        """
        Saves spherical coefficients of all iterations
        Parameters
        ----------
        basedir
            path where files will be saved
        file_name
            optional name to add to files
        save_iterations
            boolean indicating whether to save coefficients after
            each iteration. if True last coefficients is not saved separately.
        save_residual
            boolean indicating whether to save the residuals of each timestep
        """
        basedir = Path(basedir)
        basedir.mkdir(exist_ok=True)
        # save residual
        if save_residual:
            residual_frame = pd.DataFrame(self.res_iter,
                                          columns=['res x', 'res y', 'res z',
                                                   'res hor', 'res int',
                                                   'res incl', 'res decl',
                                                   'res total'])
            residual_frame.to_csv(basedir / f'{file_name}_residual.csv',
                                  sep=';')
        if save_iterations:
            np.save(basedir / f'{file_name}_all_coeff', self.unsplined_iter)
        else:
            gh_time = np.zeros((self._nm_total, len(self._t_array)))
            for gh in range(self._nm_total):
                gh_time[gh] = self.unsplined_gh(self._t_array)
            np.save(basedir / f'{file_name}_final_coeff', gh_time)

    def sweep_damping(self,
                      x0: Union[list, np.ndarray],
                      spatial_range: Union[list, np.ndarray] = [0],
                      temporal_range: Union[list, np.ndarray] = [0],
                      max_iter: int = 5,
                      prep_kwargs: dict = {},
                      save_kwargs: dict = {'save_residual': True}):
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
        max_iter
            maximum number of iterations. defaults to 5 iterations
        prep_kwargs
            keyword arguments for preparing the inversion. Optional kwargs are:
            spatial_dt, temporal_dt, temp_2nd_der, and damp_dipole
            For more info see method prepare_inversion
        save_kwargs
            optional keyword arguments for saving files. Optional kwargs are:
            basedir, file_name, save_iterations, and save_residual
            For more info see method save_spherical_coefficients

        """

        for spatial_df in tqdm(spatial_range):
            prep_kwargs['spatial_df'] = spatial_df
            for temporal_df in temporal_range:
                prep_kwargs['temporal_df'] = temporal_df
                self.run_inversion(x0, max_iter, **prep_kwargs)
                self.save_spherical_coefficients(
                    file_name=f'{spatial_df:.2e}s+{temporal_df:.2e}t',
                    **save_kwargs)
