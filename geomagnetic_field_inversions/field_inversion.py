import numpy as np
from scipy.interpolate import BSpline

from scipy.linalg import cholesky_banded, cho_solve_banded
import pandas as pd
from typing import Union, Final
from pathlib import Path
from tqdm import tqdm

from geomagnetic_field_inversions.forward_modules import (
    frechet_types,
    frechet_basis,
)
from geomagnetic_field_inversions.forward_modules.fwtools import \
    forward_obs_time
from geomagnetic_field_inversions.damping_modules import damp_matrix, damp_norm
from geomagnetic_field_inversions.tools import frechet_in_geoc
from geomagnetic_field_inversions.banded_tools.build_banded import \
    build_banded_2, build_banded_3
from geomagnetic_field_inversions.banded_tools.calc_nonzero import \
    calc_nonzero
from geomagnetic_field_inversions.banded_tools.utils import banded_mul_vec


class FieldInversion(object):
    """
    Calculates geomagnetic field coefficients based on inputted data and
    damping parameters using the approach of Korte et al.
    """

    def __init__(self,
                 t_min: float, t_max: float, t_step: float,
                 maxdegree: int = 3,
                 r_model: float = 6371.2,
                 verbose: bool = False,
                 ) -> None:
        """
        Initializes the Field Inversion class

        Parameters
        ----------
        t_min, t_max, t_step
            Sets start, end, and timestep.
        maxdegree
            maximum order for spherical harmonics model, default 3
        r_model
            where the magnetic field is modeled (km distance from core)
        verbose
            Verbosity flag, defaults to False
        """
        # basic parameters
        self._SPL_DEGREE: Final[int] = 3

        # input parameters
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.t_array = np.arange(t_min, t_max + t_step, t_step)
        # temporal knots
        self.knots = np.arange(
            t_min - self._SPL_DEGREE * t_step,
            self.t_max + (self._SPL_DEGREE + 1) * t_step,
            t_step,
        )
        # number of temporal splines
        self.nr_splines = len(self.knots) - self._SPL_DEGREE - 1

        self.maxdegree = maxdegree
        self.r_model = r_model
        self.verbose = verbose

        # initiate empty variables
        self.time = []
        self.data = []
        self.std = []
        self.unsplined_iter_gh = []
        self.idx_out = np.zeros(0)
        self.count_type = np.zeros(7)
        self.station_coord = np.zeros((0, 3))
        self.sdamp_diag = np.zeros(0)
        self.tdamp_diag = np.zeros(0)
        self.spat_norm = 0
        self.spat_type = None
        self.temp_norm = 0
        self.temp_type = None
        self.splined_gh = np.zeros(0)
        self.station_frechet = np.zeros(0)
        self.res_iter = np.zeros(0)
        self.x0 = np.zeros(0)

    @property
    def maxdegree(self):
        return self._maxdegree

    @maxdegree.setter
    def maxdegree(self, degree: int):
        # determines the maximum number of spherical coefficients
        self._nm_total = int((degree+1)**2 - 1)
        self._maxdegree = int(degree)
        self.spat_fac = np.zeros(self._nm_total)  # contains damping factors
        self.temp_fac = np.zeros(self._nm_total)
        self.matrix_ready = False

    # @profile
    def prepare_inversion(self,
                          d_inst,
                          spat_type: str = None,
                          temp_type: str = None,
                          spat_ddip: bool = False,
                          temp_ddip: bool = True
                          ) -> None:
        """
        Function to load data and prepare matrices for the inversion

        Parameters
        ----------
        d_inst
            InputData attribute containing geomagnetic data
        spat_type, temp_type
            string corresponding to the to be applied damping type
            options spatial: uniform, energy_diss, powerseries, ohmic_heating,
            smooth_core, min_ext_energy
            options temporal: min_vel, min_acc
            defaults to no damping
        spat_ddip, temp_ddip
            boolean indicating whether to damp dipole coefficients for spatial
            and temporal damping.

        Creates or modifies
        -------------------
        self.count_type
            array of length 7 recording the number of different data types
        self.idx_out
            index of data
        self.time
            time of corresponding geomagnetic datum
        self.data, self.std
            geomagnetic datum and its error
        self.station_frechet
            contains frechet matrix per location
            size= ((# stations x 3), nm_total) (floats)
        self.data_ix, self.stat_ix, self.type_ix
            contains per timestep index of datum, location, and data type
        self.spat_fac, self.temp_fac
            contains the damping elements dependent on degree
             size= nm_total (floats) (see damp_types.py)
        self.sdamp_diag
            contains diagonals of symmetric spatial damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.tdamp_diag
            contains diagonals of symmetric temporal damping matrix
            size= (nm_total x nr_splines, nm_total x nr_splines) (floats)
        self.matrix_ready
            indicates whether all matrices have been formed (boolean)
        """
        if self.verbose:
            print(d_inst)
        # order datatypes in a more straightforward way
        # line of types_sorted corresponds to index
        self.idx_out = d_inst.idx_out
        # self.idx_frech = d_inst.idx_frech
        self.idx_res = d_inst.idx_res
        self.time = d_inst.time
        # XXX why work in radians???
        self.data = d_inst.outputs.copy()
        self.data[d_inst.idx_res[5]:] = np.radians(
            d_inst.outputs[d_inst.idx_res[5]:]
        )
        self.std = d_inst.std_out.copy()
        self.std[d_inst.idx_res[5]:] = np.radians(
            d_inst.std_out[d_inst.idx_res[5]:]
        )

        # calculate frechet dx, dy, dz for all stations
        if self.verbose:
            print('Calculating Schmidt polynomials and Fréchet coefficients')
        station_coord = d_inst.loc[:, :3].copy()
        station_coord[:, :2] = np.radians(d_inst.loc[:, :2])
        # find location with geodetic coordinates
        self.station_frechet = frechet_basis(station_coord[:, :3],
                                             self._maxdegree)
        # geocentric correction
        cd, sd = d_inst.loc[:, 3], d_inst.loc[:, 4]
        dx, dz = frechet_in_geoc(
            self.station_frechet[:, 0],
            self.station_frechet[:, 2],
            cd,
            sd,
        )
        self.station_frechet[:, 0] = dx
        self.station_frechet[:, 2] = dz

        self.spatial = self.station_frechet[d_inst.loc_idx]
        # MAX: station_frechet should be related to spatial
        temporal = BSpline.design_matrix(
            d_inst.time,
            self.knots,
            self._SPL_DEGREE,
        )
        lookup_list = []
        for it in range(self.nr_splines):
            lookup_list.append(temporal[:, [it]].nonzero()[0])

        # XXX Maybe it is possible to facilitate the banded structure of
        # temporal directly
        self.temporal = np.ascontiguousarray(temporal.T.toarray())

        # Calculate indices for loop speedup.
        # def get_nleft(knots, time):
        #     return np.max(
        #         np.argwhere(knots <= time).flatten()
        #     )

        # self.nlefts = np.zeros(self.time.size, dtype=int)
        # for it, time in enumerate(self.time):
        #     self.nlefts[it] = get_nleft(self.knots, time)

        # starts, ind_list = calc_nonzero(self.temporal)
        ind_list = [None] * self.nr_splines * self.nr_splines
        starts = np.zeros(self.nr_splines * self.nr_splines + 1, dtype=int)
        starts[0] = 0
        for it in range(self.nr_splines):
            inds = np.intersect1d(lookup_list[it], lookup_list[it])
            ind_list[it * self.nr_splines + it] = inds
            starts[it * self.nr_splines + it + 1] = len(inds)
            for jt in range(it+1, self.nr_splines):
                inds = np.intersect1d(lookup_list[it], lookup_list[jt])
                idx_1 = it * self.nr_splines + jt
                idx_2 = jt * self.nr_splines + it
                ind_list[idx_1] = inds
                ind_list[idx_2] = inds
                starts[idx_1 + 1] = len(inds)
                starts[idx_2 + 1] = len(inds)

        ind_list = np.hstack(ind_list)
        self.starts = np.ascontiguousarray(np.cumsum(starts), dtype=np.int32)
        self.ind_list = np.ascontiguousarray(ind_list, dtype=np.int32)

        # Prepare damping matrices
        if spat_type is not None:
            if self.verbose:
                print('Calculating spatial damping matrix')
            self.spat_type = f's_{spat_type}'
            self.sdamp_diag, self.spat_fac = damp_matrix(
                self._maxdegree, self.nr_splines, self.t_step, self.spat_type,
                spat_ddip)

        if temp_type is not None:
            if self.verbose:
                print('Calculating temporal damping matrix')
            self.temp_type = f't_{temp_type}'
            self.tdamp_diag, self.temp_fac = damp_matrix(
                self._maxdegree, self.nr_splines, self.t_step, self.temp_type,
                temp_ddip)

        self.matrix_ready = True
        if self.verbose:
            print('Calculations finished')

    # @profile
    def run_inversion(self,
                      x0: np.ndarray,
                      spat_damp: float,
                      temp_damp: float,
                      max_iter: int = 10,
                      stop_crit: float = -np.inf,
                      path: Path = None,
                      ) -> None:
        """
        Runs the iterative inversion

        Parameters
        ----------
        x0
            starting model gaussian coefficients, should have length:
            (spherical_order + 1)^2 - 1 or
            (spherical_order + 1)^2 - 1 X nr_splines if changing through time
        spat_damp, temp_damp
            damping factor to be applied to the spatial or temporal
            damping matrix
        max_iter
            maximum amount of iterations.
        stop_crit
            stopping criterion for iterations. Iterations will stop if
            relative change of residual is less than given float (between 0-1).
        path
            path to location where to save normal_eq_splined and damp_matrix
            for calculating optional covariance and resolution matrix.
            If not provided, matrices are not saved.
            See calc_stdev in tools/core

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
        self.temp_norm, self.spat_norm
            contains temporal or spatial damping norm
        """
        if not self.matrix_ready:
            raise Exception('Matrices have not been prepared. '
                            'Please run prepare_inversion first.')
        d_matrix = spat_damp * self.sdamp_diag + temp_damp * self.tdamp_diag

        # TODO: check iteration count.
        # XXX: This leads to 2 iterations, even if maxiter = 1
        # initiate array counting residual per type
        self.res_iter = np.zeros((max_iter+1, 8))
        # initiate splined values with starting model
        if self.verbose:
            print('Setting up starting model')

        # TODO: rename stuff
        # These are the coefficients we solve for.
        self.splined_gh = np.zeros((self.nr_splines, self._nm_total))
        self.unsplined_iter_gh = []
        if x0.ndim == 1 and len(x0) == self._nm_total:
            self.x0 = x0.copy()
            self.splined_gh[:] = x0
        else:
            raise Exception(f'x0 has incorrect shape: {x0.shape}. \n'
                            f'It should have shape ({self._nm_total},)')

        spacing = self._nm_total * self._SPL_DEGREE
        # This transforms the d_matrix to the right shape. Actually,
        # the matrices should already be generated that way in the final
        # version
        C_m_inv = np.zeros(
            (
                spacing + 1,
                self.nr_splines * self._nm_total
            ),
        )

        for it in range(self._SPL_DEGREE + 1):
            C_m_inv[it * self._nm_total] = d_matrix[it].copy()

        for it in range(max_iter+1):  # start outer iteration loop
            if self.verbose:
                print(f'Start calculations iteration {it}')

            # get all predictions at once using splinebasis
            forwobs_matrix = forward_obs_time(
                self.splined_gh,
                self.spatial,
                self.temporal,
            )
            # transform the predictions into the same order as the outputs
            prediction = np.hstack(
                (
                    forwobs_matrix[0, self.idx_res[0]:self.idx_res[1]],
                    forwobs_matrix[1, self.idx_res[1]:self.idx_res[2]],
                    forwobs_matrix[2, self.idx_res[2]:self.idx_res[3]],
                    forwobs_matrix[3, self.idx_res[3]:self.idx_res[4]],
                    forwobs_matrix[4, self.idx_res[4]:self.idx_res[5]],
                    forwobs_matrix[5, self.idx_res[5]:self.idx_res[6]],
                    forwobs_matrix[6, self.idx_res[6]:self.idx_res[7]],
                )
            )
            # calculate misfit and residuals
            df = self.data - prediction
            # Consider periodicity in declinations
            df[self.idx_res[-2]:self.idx_res[-1]] += (
                2 * np.pi * (-np.pi > df[self.idx_res[-2]:self.idx_res[-1]])
                - 2 * np.pi * (df[self.idx_res[-2]:self.idx_res[-1]] > np.pi)
            )

            res = df / self.std
            for i in range(7):
                if df[self.idx_res[it]:self.idx_res[it+1]].size > 0:
                    self.res_iter[it] = np.abs(
                        df[self.idx_res[it]:self.idx_res[it+1]]
                    ).mean()
            self.res_iter[it, 7] = np.abs(res).mean()
            if self.verbose:
                print('Residual is %.2f' % self.res_iter[it, 7])

            # check if final conditions have been met
            if it > 0:
                rel_err = abs(self.res_iter[it, 7] - self.res_iter[it-1, 7]
                              ) / self.res_iter[it-1, 7]
                if stop_crit >= rel_err or it == max_iter:
                    if self.verbose:
                        print(f'Final iteration; relative error = {rel_err}')
                    break

            # solve the equations
            if self.verbose:
                print('Prepare and solve equations')
            # set up the spatial linearization / gradients
            frech_matrix = frechet_types(
                self.spatial, forwobs_matrix
            )
            frech_matrix = np.vstack(
                (
                    frech_matrix[self.idx_res[0]:self.idx_res[1], 0],
                    frech_matrix[self.idx_res[1]:self.idx_res[2], 1],
                    frech_matrix[self.idx_res[2]:self.idx_res[3], 2],
                    frech_matrix[self.idx_res[3]:self.idx_res[4], 3],
                    frech_matrix[self.idx_res[4]:self.idx_res[5], 4],
                    frech_matrix[self.idx_res[5]:self.idx_res[6], 5],
                    frech_matrix[self.idx_res[6]:self.idx_res[7], 6],
                )
            ).T
            # include the C_e^{-1/2} factor
            frech_matrix /= self.std[None, :]
            banded = build_banded_2(
                np.ascontiguousarray(frech_matrix),
                self.temporal,
                self._SPL_DEGREE,
                self.ind_list,
                self.starts,
            )
            # efficiently set up normal equations using Cython code
            # banded = build_banded_3(
            #     self.nlefts,
            #     np.ascontiguousarray(frech_matrix),
            #     self.temporal,
            #     self._SPL_DEGREE,
            # )
            # add damping to normal equations
            banded[banded.shape[0]-C_m_inv.shape[0]:] += C_m_inv
            # calculate cholesky
            chol = cholesky_banded(banded)
            # set up right hand side
            rhs = np.einsum(
                'ik,jk,k,k->ij',
                self.temporal,
                frech_matrix,
                1 / self.std,
                df,
                optimize=True,
            ).flatten()
            rhs -= banded_mul_vec(C_m_inv, self.splined_gh.flatten())
            # solve using cholesky and update solution
            self.splined_gh += cho_solve_banded((chol, False), rhs).reshape(
                self.nr_splines,
                self._nm_total,
            )
            # store iteration results as BSpline objects
            spline = BSpline(t=self.knots, c=self.splined_gh.copy(),
                             k=3, axis=0, extrapolate=False)
            self.unsplined_iter_gh.append(spline)

        # sum residuals and finish up stuff
        if self.verbose:
            print('Calculating optional spatial and temporal norms')
        tsp = self.t_array[-1] - self.t_array[0]
        if spat_damp != 0:
            self.spat_norm = damp_norm(self.spat_fac, self.splined_gh,
                                       self.spat_type, self.t_step)
            if self.verbose:
                print(f'Spatial damping norm: {np.sum(self.spat_norm) / tsp}')
        if temp_damp != 0:
            self.temp_norm = damp_norm(self.temp_fac, self.splined_gh,
                                       self.temp_type, self.t_step)
            if self.verbose:
                print(f'Temporal damping norm: {np.sum(self.temp_norm) / tsp}')

        if path is not None:
            if self.verbose:
                print('Saving matrices')
            print('Export is currently not supported')

        if self.verbose:
            print('Finished inversion')

    def save_coefficients(self,
                          basedir: Union[Path, str] = '.',
                          file_name: str = 'coeff',
                          save_iterations: bool = True,
                          save_residual: bool = False,
                          save_dampnorm: bool = False,
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
        save_dampnorm
            boolean indicating whether to save the damping norm each timestep
        """
        # save residual
        if save_residual:
            residual_frame = pd.DataFrame(
                self.res_iter, columns=['res x', 'res y', 'res z', 'res hor',
                                        'res int', 'res incl', 'res decl',
                                        'res total'])
            residual_frame.to_csv(basedir / f'{file_name}_residual.csv',
                                  sep=';')

        if save_iterations:
            all_coeff = np.zeros(
                (
                    len(self.unsplined_iter_gh),
                    len(self.t_array),
                    self._nm_total
                )
            )
            for i in range(len(self.unsplined_iter_gh)):
                all_coeff[i] = self.unsplined_iter_gh[i](self.t_array)
            np.save(basedir / f'{file_name}_all.npy', all_coeff)
        else:
            gh_time = self.unsplined_iter_gh[-1](self.t_array)
            np.save(basedir / f'{file_name}_final.npy', gh_time)
        if save_dampnorm:
            np.savez(basedir / f'{file_name}_damp.npz',
                     spat_norm=self.spat_norm,
                     temp_norm=self.temp_norm,
                     time_array=self.t_array)

    def sweep_damping(self,
                      x0: Union[list, np.ndarray],
                      spatial_range: Union[list, np.ndarray],
                      temporal_range: Union[list, np.ndarray],
                      max_iter: int = 10,
                      basedir: Path = Path().absolute(),
                      overwrite: bool = True
                      ) -> None:
        """ Sweep through damping parameters to find ideal set
        Note: Use after initiating class and running prepare_inversion

        Parameters
        ----------
        x0
            starting model gaussian coefficients, should be a float or
            as long as (spherical_order + 1)^2 - 1
        spatial_range
            array or list to vary spatial damping parameters.
        temporal_range
            array or list to vary temporal damping parameters.
        max_iter
            maximum number of iterations. defaults to 5 iterations
        basedir
            path where files will be saved
        overwrite
            boolean indicating whether to overwrite existing files with
            exactly the same damping parameters. otherwise set of damping
            parameters is skipped over in the calculations.
        """
        for spatial_df in tqdm(spatial_range):
            spat_damp = spatial_df
            for temporal_df in temporal_range:
                temp_damp = temporal_df
                if overwrite or not (basedir / f'{spatial_df:.2e}s+'
                                               f'{temporal_df:.2e}t_final.npy'
                                     ).is_file():
                    self.run_inversion(x0, spat_damp, temp_damp, max_iter)
                    self.save_coefficients(
                        file_name=f'{spatial_df:.2e}s+{temporal_df:.2e}t',
                        basedir=basedir, save_iterations=False,
                        save_residual=True, save_dampnorm=True)

    def save_to_fortran_format(self, path: Union[str, Path]) -> None:
        """ Saves the final iteration inversion result as a file in the same
        format as the Fortran code. A file format description is given  `here
        <https://sec23.git-pages.gfz-potsdam.de/korte/pymagglobal/
        overview.html#file-format-description>`_.

        Paramters
        ---------
        path
            The path where the output will be saved.
        """
        with open(path, 'w') as fh:
            fh.write(
                f'{self.t_min:.1f}  {self.t_max:.1f}  '
                f'{self._SPL_DEGREE + 1:d}\n'
            )

            fh.write(
                f'          {self.maxdegree:d}           0         '
                f'{self.nr_splines}\n'
            )
            for knot in self.knots:
                fh.write(f'{knot:0<17f}      ')
            fh.write('\n')
            for coeff in self.splined_gh.flatten():
                if 1e-1 <= coeff and coeff <= 1e5:
                    fh.write(f'{coeff:0<17f}      ')
                else:
                    fh.write(f'{coeff:.15E}      ')

    # XXX I'm not sure how to do the type hinting in this case...
    def result_to_pymagglobal(self, name: str) -> 'pymagglobal.Model':
        """ Returns the output as a [pymagglobal]_ Model instance.

        Parameters
        ----------
        name
            The model name, used internally by pymagglobal.

        Returns
        -------
            The final iteration inversion result, wrapped as a pymagglobal
            model.

        References
        ----------
        .. [pymagglobal] : Schanner, M. A.; Mauerberger, S.; Korte, M.
            "`pymagglobal - Python interface for global geomagnetic field
            models.<https://sec23.git-pages.gfz-potsdam.de/korte/
            pymagglobal/>`_", GFZ Data Services, 2020
        """
        try:
            from pymagglobal import Model

            class InvModel(Model):
                def __init__(_self):
                    _self.name = name
                    _self.t_min = self.t_min
                    _self.t_max = self.t_max
                    _self.l_max = self.maxdegree
                    _self.knots = self.knots
                    _self.coeffs = self.splined_gh
                    _self.splines = BSpline(
                        _self.knots,
                        _self.coeffs,
                        3,
                    )
                    _self.cov_splines = None

            return InvModel()

        except ImportError:
            raise ImportError(
                'pymagglobal could not be found, please install to transform '
                'the inversion result to a pymagglobal model.'
            )
