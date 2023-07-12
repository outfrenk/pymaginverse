import numpy as np
from scipy.integrate import newton_cotes
from scipy.interpolate import BSpline, interp1d
from scipy.linalg import pinv
import pandas as pd
import pyshtools as pysh
from typing import Union, Literal
from pathlib import Path
from tqdm import tqdm

from geomagnetic_field_inversions.data_prep import StationData
_DampingMethods = Literal['spatial_G', 'temporal']


# TODO: understand lat_geod_geoc function
def lat_geod_geoc(lat: Union[float, np.ndarray, list],
                  h: Union[float, np.ndarray, list] = 0):
    """ Transforms the geodetic latitude to spherical coordinates
    Additionally, calculates the new radius and conversion factors
    Note: copied from old Fortran code

    Parameters
    ----------
    lat
        to be converted latitude in radians
    h
        height above geoid

    Returns
    -------
    new_lat
        converted latitude
    new_rad
        recalculated radius
    cd
        conversion factor for calculating geodetic magnetic x,z components.
        If geocentric, cd is 1
    sd
        conversion factor for calculating geodetic magnetic x,z components.
        If geocentric, sd is 0
    """
    b1 = 40680925.0
    b2 = 40408585.0

    one = b1 * np.cos(lat) ** 2
    two = b2 * np.sin(lat) ** 2
    three = one + two
    four = np.sqrt(three)
    new_rad = np.sqrt(h * (h + 2 * four) + (b1 * one + b2 * two) / three)
    cd = (h + four) / new_rad
    sd = (b1 - b2) / four * np.sin(lat) * np.cos(lat) / new_rad
    sinth = np.sin(lat) * cd - np.cos(lat) * sd
    costh = np.cos(lat) * cd + np.sin(lat) * sd
    new_lat = np.arctan2(sinth, costh)
    return new_lat, new_rad, cd, sd


def mag_geoc_geod(mx: Union[list, np.ndarray, float],
                  mz: Union[list, np.ndarray, float],
                  cd: Union[list, np.ndarray, float],
                  sd: Union[list, np.ndarray, float]):
    """ Transforms magnetic components from geocentric
    to correct geodetic value

    Parameters
    ----------
    mx, mz
        to be converted magnetic vectors
    cd, sd
        conversion factors for calculating magnetic x, z components.
        cd = 1 indicates perfect spherical earth
    """
    mx_new = mx * cd + mz * sd
    mz_new = mz * cd - mx * sd
    return mx_new, mz_new


class FieldInversion:
    """
    Calculates geomagnetic field coefficients based on inputted data and
    damping parameters using the approach of Korte et al. (????)
    """

    def __init__(self,
                 time_array: Union[list, np.ndarray],
                 maxdegree: int = 3,
                 spl_order: int = 3,
                 r_model: float = 6371.2,
                 r_earth: float = 6371.2,
                 cmb_earth: float = 3485.0,
                 geodetic: bool = True,
                 verbose: bool = False):
        """
        Initializes the Field Inversion class
        
        Parameters
        ----------
        time_array
            Sets timearray for the inversion in yr. Should be ascending
        maxdegree
            maximum order for spherical harmonics model, default 3
        spl_order
            order of splines used for relating timesteps together, default 3
        r_model
            where the magnetic field is modeled (km distance from core)
        r_earth
            radius of the earth (in km)
        cmb_earth
            radius core mantle boundary (in km)
        geodetic
            boolean specifying whether to use a geodetic coordinate frame. If
            True, geodetic coordinate frame is used and recalculated into a
            geocentric one. Otherwise, a geocentric frame is used.
            Default is geodetic (True)
        verbose
            Verbosity flag, defaults to False
        """
        self.t_array = np.sort(time_array)
        self.maxdegree = maxdegree
        self.spl_order = spl_order
        self.r_model = r_model
        self.r_earth = r_earth
        self.cmb_earth = cmb_earth
        self.geodetic = geodetic
        self.verbose = verbose
        # initiate blank data, error, type, coordinate, conversion arrays, etc.
        self.schmidt_P = np.empty(0)
        self.schmidt_dP = np.empty(0)
        self.data_array = np.zeros((0, len(time_array)))
        self.error_array = np.zeros((0, len(time_array)))
        self.time_cover = np.zeros((0, len(time_array)))
        self.types = []
        self.station_coord = np.zeros((0, 3))
        self.gcgd_conv = np.zeros((0, 2))
        self.spatial_damp_matrix = np.empty(0)
        self.temporal_damp_matrix = np.empty(0)
        self.splined_gh = np.empty(0)
        self.unsplined_gh = np.empty(0)
        self.time_knots = np.empty(0)
        self.nr_splines = np.empty(0)
        self.station_frechet = np.empty(0)
        self.res_iter = np.empty(0)
        self.unsplined_iter = np.empty(0)

    @property
    def maxdegree(self):
        return self._maxdegree

    @maxdegree.setter
    def maxdegree(self, degree: int):
        # determines the maximum number of spherical coefficients
        self._nm_total = int((degree + 1)**2 - 1)
        self._maxdegree = int(degree)
        self.matrix_ready = False

    @property
    def t_array(self):
        return self._t_array

    @t_array.setter
    def t_array(self, array: Union[list, np.ndarray]):
        if len(array) == 1:
            self._t_step = 0
            self._t_array = array
        else:
            self._t_step = array[1] - array[0]
            self._t_array = array
            for i in range(len(self._t_array) - 1):
                if self._t_array[i+1] - self._t_array[i] != self._t_step:
                    raise Exception("Time vector has different timesteps."
                                    " Redefine vector with same timestep")
        self.matrix_ready = False

    @property
    def spl_order(self):
        return self._spl_order

    @spl_order.setter
    def spl_order(self, spline_order: int):
        self._spl_order = spline_order
        self._bspline = BSpline.basis_element(np.arange(spline_order + 2),
                                              extrapolate=False)
        self.matrix_ready = False

    def add_data(self,
                 data_class: StationData,
                 error_interp: str = 'linear'):
        """
        Adds data generated by the Station_data class
        
        Parameters
        ----------
        data_class
            instance of the Station_data class. Only added if it matches the
            time_array set in __init__
        error_interp
            string specifying interpolation of inputted error to time_array
        """
        # translation datatypes
        typedict = {"x": 0, "y": 1, "z": 2, "hor": 3,
                    "int": 4, "inc": 5, "dec": 6}
        if isinstance(data_class, StationData):
            data_entry = np.zeros((len(data_class.types), len(self._t_array)))
            # set unknown errors to one to avoid divide by zero error frechet
            error_entry = np.ones((len(data_class.types), len(self._t_array)))
            # time_cover indicates timerange of data (used in Fréchet)
            time_cover = np.zeros((len(data_class.types), len(self._t_array)))
            types_entry = []
            for c, types in enumerate(data_class.types):
                # check coverage data timevector begin
                arg_min, arg_max = 0, len(self._t_array)
                if data_class.data[c][0][0] > self._t_array[0]:
                    if self.verbose:
                        print(f'{types} of {data_class.__name__} not covering start time')
                    if data_class.data[c][0][0] > self._t_array[-1]:
                        raise Exception(f'{types} of {data_class.__name__} does not cover'
                                        ' any timestep of timevector')
                    args = np.argsort(np.abs(data_class.data[c][0][0]
                                             - self._t_array))
                    if args[0] > args[1]:
                        arg_min = args[0]
                    else:
                        arg_min = args[1]

                # check coverage data timevector end
                if data_class.data[c][0][-1] < self._t_array[-1]:
                    if self.verbose:
                        print(f'{types} of {data_class.__name__} not covering end time')
                    if data_class.data[c][0][-1] < self._t_array[0]:
                        raise Exception(f'{types} of {data_class.__name__} does not cover'
                                        ' any timestep of timevector')
                    args = np.argsort(np.abs(data_class.data[c][0][-1]
                                             - self._t_array))
                    if args[0] < args[1]:
                        arg_max = args[0] + 1
                    else:
                        arg_max = args[1] + 1
                time_cover[c, arg_min:arg_max] = 1
                # TODO: change data entry into geocentric frame (small)
                if self.verbose:
                    print(f'Adding {types}-type')
                if types == 'inc':
                    temp = data_class.fit_data[c](self._t_array[arg_min:
                                                                arg_max])
                    data_entry[c, arg_min:
                                  arg_max] = np.radians(temp)
                elif types == 'dec':
                    temp = data_class.fit_data[c](self._t_array[arg_min:
                                                                arg_max])
                    data_entry[c, arg_min:
                                  arg_max] = np.radians(temp)
                else:
                    data_entry[c, arg_min:arg_max] = data_class.fit_data[c](
                        self._t_array[arg_min:arg_max])
                # sample errors for time_array
                f = interp1d(data_class.data[c][0],
                             data_class.data[c][2],
                             kind=error_interp)
                error_entry[c, arg_min:arg_max] = f(self._t_array[arg_min:
                                                                  arg_max])

                types_entry.append(typedict[types])

            # change coordinates to geocentric if required
            if self.geodetic:
                if self.verbose:
                    print(f'Coordinates are geodetic,'
                          ' translating to geocentric coordinates.')
                lat_geoc, r_geoc, cd, sd = \
                    lat_geod_geoc(np.radians(data_class.lat))
                station_entry = np.array([0.5 * np.pi - lat_geoc,
                                          np.radians(data_class.lon),
                                          r_geoc])
            else:
                if self.verbose:
                    print(f'Coordinates are geocentric,'
                          ' no translation required.')
                cd = 1.
                sd = 0.
                station_entry = np.array([np.radians(90 - data_class.lat), 
                                          np.radians(data_class.lon),
                                          self.r_earth])

            # add data to attributes of the class if all is fine
            if self.verbose:
                print(f'Data of {data_class.__name__} is added to class')
            self.data_array = np.vstack((self.data_array, data_entry))
            self.error_array = np.vstack((self.error_array, error_entry))
            self.time_cover = np.vstack((self.time_cover, time_cover))
            self.types.append(types_entry)
            self.station_coord = np.vstack((self.station_coord, station_entry))
            self.gcgd_conv = np.vstack((self.gcgd_conv, np.array([cd, sd])))
        else:
            raise Exception('data_class is not an instance of Station_Data')

    def prepare_inversion(self,
                          spatial_df: float,
                          temporal_df: float,
                          damp_dipole: bool = False):
        """ 
        Function to prepare matrices for the inversion
        
        Parameters
        ----------
        spatial_df
            spatial damping factor applied to the matrix, default 3e-10
        temporal_df
            temporal damping factor applied to the matrix, default 1e-2
        damp_dipole
            boolean indicating whether to damp dipole coefficients or not.
            Default is set to False.

        """
        # check data and model space
        assert self._nm_total <= len(self.data_array), \
            'The spherical order of the model is too high,' \
            f' decrease maxdegree from {self._maxdegree} to a lower value.'

        # calculate schmidt polynomials and frechet x,y,z for all stations
        if self.verbose:
            print('Calculating Schmidt polynomials and Fréchet coefficients')
        self.schmidt_P = np.zeros((len(self.station_coord),
                                   int((self.maxdegree + 1)
                                       * (self.maxdegree + 2) / 2)))
        self.schmidt_dP = np.zeros((len(self.station_coord),
                                    int((self.maxdegree + 1)
                                        * (self.maxdegree + 2) / 2)))
        for i, coord in enumerate(self.station_coord):
            self.schmidt_P[i], self.schmidt_dP[i] = \
                pysh.legendre.PlmSchmidt_d1(self.maxdegree, np.cos(coord[0]))
            self.schmidt_dP[i] *= -np.sin(coord[0])
        self.station_frechet = np.zeros((len(self.station_coord),
                                         3 * self._nm_total))
        counter = 0
        for n in range(1, self._maxdegree + 1):
            index = int(n * (n + 1) / 2)
            mult_factor = (self.r_earth / self.station_coord[:, 2]) ** (n + 1)
            # dx, dy, dz in one row separated by self._nm_total
            self.station_frechet[:, counter] =\
                mult_factor * self.schmidt_dP[:, index]
            self.station_frechet[:, counter+self._nm_total] = 0
            self.station_frechet[:, counter+2*self._nm_total] =\
                -mult_factor * (n + 1) * self.schmidt_P[:, index]
            counter += 1
            for m in range(1, n + 1):
                # First the g-elements
                self.station_frechet[:, counter] =\
                    mult_factor * self.schmidt_dP[:, index+m]\
                    * np.cos(m * self.station_coord[:, 1])
                self.station_frechet[:, counter+self._nm_total] = \
                    m / np.sin(self.station_coord[:, 0]) * mult_factor\
                    * np.sin(m * self.station_coord[:, 1])\
                    * self.schmidt_P[:, index+m]
                self.station_frechet[:, counter+2*self._nm_total] =\
                    -mult_factor * (n + 1) * self.schmidt_P[:, index+m]\
                    * np.cos(m * self.station_coord[:, 1])
                counter += 1
                # Now the h-elements
                self.station_frechet[:, counter] =\
                    mult_factor * self.schmidt_dP[:, index+m]\
                    * np.sin(m * self.station_coord[:, 1])
                self.station_frechet[:, counter+self._nm_total] =\
                    -m / np.sin(self.station_coord[:, 0]) * mult_factor\
                    * np.cos(m * self.station_coord[:, 1])\
                    * self.schmidt_P[:, index+m]
                self.station_frechet[:, counter+2*self._nm_total] =\
                    -mult_factor * (n + 1) * self.schmidt_P[:, index+m]\
                    * np.sin(m * self.station_coord[:, 1])
                counter += 1
        # geocentric correction
        dx = self.station_frechet[:, 2*self._nm_total:]\
            * self.gcgd_conv[:, 1, np.newaxis]\
            + self.station_frechet[:, :self._nm_total]\
            * self.gcgd_conv[:, 0, np.newaxis]
        dz = self.station_frechet[:, 2*self._nm_total:]\
            * self.gcgd_conv[:, 0, np.newaxis]\
            - self.station_frechet[:, :self._nm_total]\
            * self.gcgd_conv[:, 1, np.newaxis]
        self.station_frechet[:, :self._nm_total] = dx
        self.station_frechet[:, 2*self._nm_total:] = dz

        if self.verbose:
            print('Setting up splines and timeknots')
        # TODO: check physical meaning nr_splines
        # number of temporal splines, convolve spline-order with time array
        self.nr_splines = len(self._t_array) + self._spl_order - 1

        # location of timeknots
        self.time_knots = np.linspace(
            self._t_array[0] - self._spl_order * self._t_step,
            self._t_array[-1] + self._spl_order * self._t_step,
            num=len(self._t_array) + 2 * self._spl_order)

        # Prepare damping matrices
        # TODO: verify physical meaning damping
        self.spatial_damp_matrix = np.zeros(
            (self._nm_total * self.nr_splines,
             self._nm_total * self.nr_splines))
        if spatial_df != 0 and self._t_step != 0:
            if self.verbose:
                print('Setting up spatial damping matrix')
            spatial_damp = self.damping('spatial_G', damp_dipole)
            for j in range(self.nr_splines):  # loop through splines with j
                for k in range(self.nr_splines):  # and loop with k
                    if abs(j - k) <= self._spl_order:
                        low = max(j, k, self._spl_order)
                        high = min(j + self._spl_order,
                                   k + self._spl_order,
                                   self.nr_splines - 1)
                        s_damper = self.integr_nc_spl(j, k, low, high)
                        # multiply factor with damping factors in diag matrix
                        s_damp_coef = s_damper * np.diag(spatial_damp)
                        self.spatial_damp_matrix[
                            j * self._nm_total:(j + 1) * self._nm_total,
                            k * self._nm_total:(k + 1) * self._nm_total
                        ] = s_damp_coef * spatial_df

        self.temporal_damp_matrix = np.zeros(
            (self._nm_total * self.nr_splines,
             self._nm_total * self.nr_splines))
        if temporal_df != 0 and self._t_step != 0:
            if self.verbose:
                print('Setting up temporal damping matrix')
            temporal_damp = self.damping('temporal', True)
            for j in range(self.nr_splines):  # loop through splines with j
                for k in range(self.nr_splines):  # and loop with k
                    if abs(j - k) <= self._spl_order:
                        low = max(j, k, self._spl_order)
                        high = min(j + self._spl_order,
                                   k + self._spl_order,
                                   self.nr_splines - 1)
                        t_damper = self.temp_nc_spl(j, k, low, high,
                                                    temp_order=1)
                        t_damp_coef = t_damper * np.diag(temporal_damp)
                        self.temporal_damp_matrix[
                            j * self._nm_total:(j + 1) * self._nm_total,
                            k * self._nm_total:(k + 1) * self._nm_total
                        ] = t_damp_coef * temporal_df

        self.matrix_ready = True

    def run_inversion(self,
                      x0: Union[np.ndarray, list],
                      max_iter: int = 5,
                      **prep_kwargs):
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
            has not been run yet. Requires at least spatial_df and
            temporal_df. See self.prepare_inversion for more information.

        """
        # TODO: add uncertainty
        if len(self._t_array) == 1:
            raise Exception('Switch to function "run_inversion_notime" to'
                  ' execute calculations for one timestep')
        if self.matrix_ready is False:
            if self.verbose:
                print('Preparing matrices for iterative inversion')
            self.prepare_inversion(**prep_kwargs)

        self.res_iter = np.zeros((max_iter+1, 8))
        self.unsplined_iter = np.zeros((max_iter,
                                        self._nm_total * len(self._t_array)))
        # initiate splined values with starting model
        if self.verbose:
            print('Setting up starting model')
        assert len(x0) == self._nm_total, \
            f'x0 has incorrect shape: {len(x0)},'\
            f' it should be: {self._nm_total}'
        self.splined_gh = np.zeros((self.nr_splines, self._nm_total))
        self.splined_gh[:] = x0
        for iteration in range(max_iter):
            if self.verbose:
                print(f'Start iteration {iteration+1}')
            count_all = np.zeros(7)
            rhs_matrix = np.zeros((len(self._t_array), self._nm_total))
            normal_eq_splined = np.zeros((self._nm_total * self.nr_splines,
                                          self._nm_total * self.nr_splines))

            rhs_spatial_damp = -1 * np.matmul(self.spatial_damp_matrix,
                                              self.splined_gh.flatten())
            rhs_temporal_damp = -1 * np.matmul(self.temporal_damp_matrix,
                                               self.splined_gh.flatten())
            gh_timesteps = BSpline(c=self.splined_gh, t=self.time_knots,
                                   k=self._spl_order, axis=0,
                                   extrapolate=False)(self._t_array)

            for t in range(len(self._t_array)):
                # Calculate the forward observation
                forw_obs, frechet_matrix, res_obs, count =\
                    self.forward_frechet(gh_timesteps[t], t, iteration)
                count_all += count
                # save residual
                self.res_iter[iteration, 7] += np.sum(
                    (res_obs / self.error_array[:, t])**2)
                # multiply the 'right hand side' and apply covariance matrix
                rhs_matrix[t, :] =\
                    np.matmul(frechet_matrix.T / self.error_array[:, t],
                              res_obs / self.error_array[:, t])
                # Apply B-Splines straight away (much easier)
                for j in range(self._spl_order):
                    for k in range(self._spl_order):
                        normal_eq_splined[
                            (t+j) * self._nm_total:(t+j+1) * self._nm_total,
                            (t+k) * self._nm_total:(t+k+1) * self._nm_total
                        ] += np.matmul(frechet_matrix.T * self._bspline(j + 1)
                                       / self.error_array[:, t]**2,
                                       frechet_matrix * self._bspline(k + 1))
            for i in range(7):
                if count_all[i] != 0:
                    self.res_iter[iteration, i] = np.sqrt(
                        self.res_iter[iteration, i] / count_all[i])
            self.res_iter[iteration, 7] = np.sqrt(self.res_iter[iteration, 7]
                                                  / np.sum(count_all))
            rhs_splined = np.zeros(self.nr_splines * self._nm_total)
            for i in range(self._nm_total):
                rhs_splined[i::self._nm_total] =\
                    np.convolve(rhs_matrix[:, i],
                                self._bspline(np.arange(1, self._spl_order+1)))
            # add spatial and temporal damping to the matrix and vector
            normal_eq_splined +=\
                self.spatial_damp_matrix + self.temporal_damp_matrix
            rhs_splined += rhs_spatial_damp + rhs_temporal_damp

            # solve the equations
            update = np.matmul(pinv(normal_eq_splined), rhs_splined)
            self.splined_gh = (self.splined_gh.flatten() + update).reshape(
                self.nr_splines, self._nm_total)
            self.unsplined_gh = np.zeros((len(self._t_array), self._nm_total))
            # cut of the sides that do not have physical meaning
            for gh in range(self._nm_total):
                self.unsplined_gh[:, gh] = np.convolve(
                    self.splined_gh[:, gh],
                    self._bspline(np.arange(1, self._spl_order+1))
                )[self._spl_order - 1:-(self._spl_order - 1)]
            self.unsplined_iter[iteration, :] = self.unsplined_gh.flatten()
            if self.verbose:
                print('Residual is %.2f' % self.res_iter[iteration, 7])
            # residual after last iteration
            if iteration == max_iter - 1:
                if self.verbose:
                    print('Calculate residual last iteration')
                gh_timesteps = BSpline(c=self.splined_gh, t=self.time_knots,
                                       k=self._spl_order, axis=0,
                                       extrapolate=False)(self._t_array)
                count_all = np.zeros(7)
                for t in range(len(self._t_array)):
                    forw_obs, frechet_matrix, res_obs, count = \
                        self.forward_frechet(gh_timesteps[t], t, iteration+1)
                    count_all += count
                    self.res_iter[iteration+1, 7] += np.sum(
                        (res_obs / self.error_array[:, t]) ** 2)
                for i in range(7):
                    if count_all[i] != 0:
                        self.res_iter[iteration+1, i] = np.sqrt(
                                self.res_iter[iteration+1, i] / count_all[i])
                self.res_iter[iteration+1, 7] =\
                    np.sqrt(self.res_iter[iteration+1, 7] / np.sum(count_all))
                if self.verbose:
                    print('Residual is %.2f' % self.res_iter[iteration+1, 7])

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
        for iteration in range(max_iter):
            if self.verbose:
                print(f'Start iteration {iteration + 1}')
            count_all = np.zeros(7)

            # Calculate the forward observation
            forw_obs, frechet_matrix, res_obs, count = \
                self.forward_frechet(self.unsplined_gh, 0, iteration)
            count_all += count
            # save residual
            self.res_iter[iteration, 7] += np.sum(
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
                    self.res_iter[iteration, i] = np.sqrt(
                        self.res_iter[iteration, i] / count_all[i])
            self.res_iter[iteration, 7] = np.sqrt(self.res_iter[iteration, 7]
                                                  / np.sum(count_all))

            # solve the equations
            update = np.matmul(pinv(normal_eq_unsplined), rhs_unsplined)
            self.unsplined_gh += update
            # cut of the sides that do not have physical meaning
            self.unsplined_iter[iteration, :] = self.unsplined_gh
            if self.verbose:
                print('Residual is %.2f' % self.res_iter[iteration, 7])
            # residual after last iteration
            if iteration == max_iter - 1:
                if self.verbose:
                    print('Calculate residual last iteration')
                count_all = np.zeros(7)
                forw_obs, frechet_matrix, res_obs, count = \
                    self.forward_frechet(self.unsplined_gh, 0, iteration + 1)
                count_all += count
                self.res_iter[iteration + 1, 7] += np.sum(
                    (res_obs / self.error_array[:, 0]) ** 2)
                for i in range(7):
                    if count_all[i] != 0:
                        self.res_iter[iteration + 1, i] = np.sqrt(
                            self.res_iter[iteration + 1, i] / count_all[i])
                self.res_iter[iteration + 1, 7] = \
                    np.sqrt(
                        self.res_iter[iteration + 1, 7] / np.sum(count_all))
                if self.verbose:
                    print('Residual is %.2f' % self.res_iter[iteration + 1, 7])

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
            np.save(basedir / f'{file_name}_final_coeff', self.unsplined_gh)

    def damping(self,
                damp_type: _DampingMethods,
                damp_dipole: bool = False):
        """ Creates spatial or temporal damping factors

        Parameters
        ----------
        damp_type
            style of damping according. Options are:
            spatial_G -> spatial damping of heat flow at
                         the core mantle boundary (Gubbins et al.)
            temporal -> minimize integral of magnetic field
                        squared over surface at core mantle boundary
        damp_dipole
            if False, damping is not applied to dipole coefficients (first 3).
            If True, dipole coefficients are damped.

        Returns
        -------
        damp_array
            contains damping value for gaussian coefficients. Essentially
            diagonal values of the damping matrix

        """
        damp_array = np.zeros((self._maxdegree + 1) ** 2 - 1)
        damp = np.zeros(self._maxdegree)
        counter = 0
        # Spatial damping according to Gubbins with 2l+3
        if damp_type == 'spatial_G':
            for degree in range(1, self._maxdegree + 1):  # starts at one!
                damp[degree - 1] = (degree + 1) * (2 * degree + 1) *\
                                   (2 * degree + 3) / degree *\
                                   (self.r_earth / self.cmb_earth) **\
                                   (2 * degree + 3)
                damp[degree - 1] = damp[degree - 1] * 4 * np.pi

        elif damp_type == 'temporal':  # weird function
            for degree in range(1, self._maxdegree + 1):
                damp[degree - 1] = (degree + 1) ** 2 / (2 * degree + 1) * \
                                   ((self.r_earth / self.cmb_earth) **
                                    (2 * degree + 4))

        else:
            raise Exception(f'Damping type {damp_type} not found. Exiting...')

        for degree in range(1, self._maxdegree + 1):
            for order in range(degree + 1):  # order should start at zero
                if damp_dipole is False and degree == 1:
                    if order == 0:
                        damp_array[counter] = 0
                        counter += 1
                    if order == 1:
                        damp_array[counter] = 0
                        counter += 1
                        damp_array[counter] = 0
                        counter += 1

                elif order == 0:
                    damp_array[counter] = damp[degree - 1]
                    counter += 1
                else:
                    damp_array[counter] = damp[degree - 1]
                    counter += 1
                    damp_array[counter] = damp[degree - 1]
                    counter += 1
        return damp_array

    # TODO: understand integration functions
    def integr_nc_spl(self,
                      j: int,
                      k: int,
                      low: int,
                      high: int,
                      newcot_order: int = 6):
        """ Integrates the splines over time using Newton-Cotes

        Parameters
        ----------
        j, k
            value between 0 and nr_splines. Indicates
            which spline to use. if j_order = 3 and
            5 splines then
            spl | j times data at timestep t
            ____|_______________________________
            0   | 0 at t=0
            1   | 1 at t=0 + 0 at t=1
            2   | 2 at t=0 + 1 at t=1 + 0 at t=2
            3   | 2 at t=1 + 1 at t=2 + 0 at t=3
            4   | 2 at t=2 + 1 at t=3 + 0 at t=4

        low, high
            indices of time_knots indicating the time over which to integrate
        newcot_order
            order+1 steps used in the Newton-Cotes integral

        Returns
        -------
        int_prod
            integration product of splines

        """

        newcot, error = newton_cotes(newcot_order)  # get the weigh factor
        bspline_matrix = np.zeros((self._spl_order + 1, newcot_order + 1))
        bspline = BSpline.basis_element(np.arange(self._spl_order + 2),
                                        extrapolate=False)
        # necessary to get sum = 1 for weigh factors
        dt = self._t_step / newcot_order

        for i in range(self._spl_order + 1):
            # create correct splines to convolve with!
            bspline_matrix[i] = bspline(
                np.linspace(i, i + 1, newcot_order + 1))
        int_prod = 0
        # integrate for time 'spend' with the spline combination
        for t in range(int(low), int(high + 1)):
            # 'some kind of' convolution
            int_prod += np.sum(
                newcot * bspline_matrix[(j + self._spl_order) - t][::-1]
                * bspline_matrix[(k + self._spl_order) - t][::-1]) * dt
        return int_prod

    def temp_nc_spl(self,
                    j: int,
                    k: int,
                    low: int,
                    high: int,
                    temp_order: int = 1):
        """ Integrates the splines over time using Newton-Cotes

        Parameters
        ----------
        j, k
            value between 0 and nr_of_splines. Indicates which spline to use
        low, high
            indices of time_knots indicating the time over which to integrate
        temp_order
            order of B-Spline used for temporal integration

        Returns
        -------
        int_prod
            integration product of splines

        """
        # TODO: research influence temp_order on temporal integration
        #              v
        newcot_order = 2
        newcot, error = newton_cotes(newcot_order)  # get the weigh factor
        bspline_matrix = np.zeros((temp_order + 1, newcot_order + 1))
        bspline = BSpline.basis_element(np.arange(temp_order + 2),
                                        extrapolate=False)
        dt = self._t_step / newcot_order
        for i in range(temp_order + 1):
            # create correct splines to convolve with!
            bspline_matrix[i] = bspline(
                np.linspace(i, i + 1, newcot_order + 1)
            )[::-1]
        coeff = np.zeros(3)
        coeff[0] = 1 / self._t_step**2
        coeff[1] = -2 / self._t_step**2
        coeff[2] = 1 / self._t_step**2
        int_prod = 0
        # integrate for time 'spend' with the spline combination
        for t in range(low, high + 1):
            iint_prod = 0
            for ndel in range(newcot_order + 1):
                spl = np.zeros(len(self.time_knots))
                spl[t] = coeff[0] * bspline_matrix[1, ndel]
                spl[t - 1] = coeff[0] * bspline_matrix[0, ndel] \
                    + coeff[1] * bspline_matrix[1, ndel]
                spl[t - 2] = coeff[1] * bspline_matrix[0, ndel]\
                    + coeff[2] * bspline_matrix[1, ndel]
                spl[t - 3] = coeff[2] * bspline_matrix[0, ndel]
                iint_prod += newcot[ndel] * spl[j] * spl[k]
            int_prod += iint_prod * dt
        return int_prod

    def forward_frechet(self,
                        coeff: Union[list, np.ndarray],
                        t: int,
                        iteration: int):
        """ Calculates forward observations, frechet matrix, and residual

        Parameters
        ----------
        coeff
            contains spherical harmonics coefficients of iteration
        t
            time index at forward calculation
        iteration
            iterationnumber in run_inversion

        Returns
        -------
        forw_obs
             forward modeled data
        frechet_matrix
            frechet matrix corresponding to datatypes
        res_obs
            residual of the observation (data minus model)
        count
            count of different datatypes
        """
        # TODO: change function to be compatible with no data
        forw_obs = np.zeros(len(self.data_array))
        frechet_matrix = np.zeros((len(self.data_array), self._nm_total))
        res_obs = np.zeros(len(self.data_array))
        count = np.zeros(7)
        forw = np.zeros(7)
        frechet = np.zeros((7, self._nm_total))
        counter = 0
        for i, station_types in enumerate(self.types):
            # calculate the possible observations and frechet matrix
            forw[0] = np.matmul(self.station_frechet[i, :self._nm_total],
                                coeff)  # x
            frechet[0] = self.station_frechet[i, :self._nm_total]
            forw[1] = np.matmul(self.station_frechet[i, self._nm_total:
                                                     2*self._nm_total],
                                coeff)  # y
            frechet[1] = self.station_frechet[i, self._nm_total:
                                              2*self._nm_total]
            forw[2] = np.matmul(self.station_frechet[i, 2*self._nm_total:],
                                coeff)  # z
            frechet[2] = self.station_frechet[i, 2*self._nm_total:]
            forw[3] = np.sqrt(forw[0]**2 + forw[1]**2)  # hor
            frechet[3] = (forw[0]*frechet[0] + forw[1]*frechet[1]) / forw[3]
            forw[4] = np.linalg.norm(forw[0:3])  # intens
            frechet[4] = (forw[3]*frechet[3] + forw[2]*frechet[2]) / forw[4]
            forw[5] = np.arcsin(forw[2] / forw[4])  # incl
            frechet[5] = (forw[3]*frechet[2] - forw[2]*frechet[3]) / forw[4]**2
            forw[6] = np.arctan2(forw[1], forw[0])  # decl
            frechet[6] = (forw[0]*frechet[1] - forw[1]*frechet[0]) / forw[3]**2
            # print(forw)
            # fill arrays and matrices with required datatype
            for j in station_types:
                count[j] += 1
                forw_obs[counter] = forw[j]
                frechet_matrix[counter, :] = frechet[j]
                res_obs[counter] = self.data_array[counter, t] - forw[j]
                if j == 5 or j == 6:  # inclination or declination
                    while res_obs[counter] > np.pi:
                        res_obs[counter] -= 2 * np.pi
                    while res_obs[counter] < -np.pi:
                        res_obs[counter] += 2 * np.pi
                # if no data, set to zero, otherwise one
                if self.time_cover[counter, t] == 0:
                    res_obs[counter] = 0
                    forw_obs[counter] = 0
                    frechet_matrix[counter, :] = 0
                    count[j] -= 1
                self.res_iter[iteration, j] += (
                    res_obs[counter] / self.error_array[counter, t])**2
                counter += 1
        return forw_obs, frechet_matrix, res_obs, count

    def sweep_damping(self,
                      x0: Union[list, np.ndarray],
                      spatial_range: Union[list, np.ndarray] = [0],
                      temporal_range: Union[list, np.ndarray] = [0],
                      damp_dipole=False,
                      inv_kwargs={'max_iter': 5},
                      save_kwargs={'basedir': '.', 'save_residual': True}):
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
        damp_dipole
            if False, damping is not applied to dipole coefficients (first 3).
            If True, dipole coefficients are damped.
        inv_kwargs
            keyword arguments for running the inversion: number of iterations
            (max_iter)
        save_kwargs
            keyword arguments for saving files: directory to save files
            (basedir) and whether to save residuals (save_residual).
            For more info see method save_spherical_coefficients

        """

        for spatial_df in tqdm(spatial_range):
            for temporal_df in temporal_range:
                self.prepare_inversion(spatial_df=spatial_df,
                                       temporal_df=temporal_df,
                                       damp_dipole=damp_dipole)
                self.run_inversion(x0=x0, **inv_kwargs)
                self.save_spherical_coefficients(
                    file_name=f'{spatial_df:.2e}s+{temporal_df:.2e}t',
                    **save_kwargs)
