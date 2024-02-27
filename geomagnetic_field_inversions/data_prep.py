import numpy as np
import pandas as pd
from pathlib import Path
from warnings import warn
from typing import Union, Optional
from .tools import latrad_in_geoc


class InputData(object):
    """Class that contains geomagnetic data and locations in order to perform
    a geomagnetic field inversion

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing input data. It should have the following
        columns:
            lat : the record latitude
            lon : the record longitude
            h : the height of the records location above the reference
            ellipsoid
            geoc : 1 or 0, indicating whether the above coords are given in a
            geocentric reference frame
            t : the record date as a calendar year
            dt : error in the record date in calendar years
            and at least one of the following
            X and dX : value and error of the magnetic field X-component
            Y and dY : value and error of the magnetic field Y-component
            Z and dZ : value and error of the magnetic field Z-component
            H and dH : value of the magnetic field horizontal intensity
            F and dF : value of the magnetic field intensity
            I and dI : value of the magnetic field inclination
            D and dD : value of the magnetic field declination

    Attributes
    ----------
    data : DataFrame
        The DataFrame containing all data.
    n_inp : int
        The number of inputs, i.e. rows of the DataFrame
    loc : array
        The unique inputs of data locations (lat, lon, radius, cd, and sd)
    loc_idx: array
        The array connecting index loc to data input
    time: array
        The array containing dated time of magnetic record
    idx_X, idx_Y, idx_Z, idx_H, idx_D, idx_I, idx_F : index
        The indices of X, Y, Z, H, dec, inc, and int records in the DataFrame.
    n_out : int
        The number of outputs, i.e. the total number of individual data points.
    outputs : array
        The outputs (order is X, Y, Z, H, dec, inc, int)
    std_out : array
        The output-errors.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = self._check_consistency(data.copy())
        self.data = self.geoc_transform_loc(self.data)
        self.compile_data()

    def _check_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Check if a dataframe is consistent with the data model. """
        # this evades errors due to missing column names
        data = data.reindex(
            data.columns.union(
                [
                    'lat', 'lon', 'h', 't', 'X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                    'H', 'dH', 'D', 'dD', 'I', 'dI', 'F', 'dF',
                    'geoc', 'geoc_colat', 'geoc_rad', 'cd', 'sd'
                ],
                sort=False,
            ),
            axis=1,
        )
        for name in ['lat', 'lon', 't']:
            idx = data.query(f'{name} != {name}').index.to_numpy()
            if idx.size > 0:
                raise Exception(
                    f"Records with indices {idx} have no {name} information."
                )
        # check latitude and longitude
        llcond = (
            (abs(data['lat']) > 90)
            | ((data['lon'] > 360) | (data['lon'] < -180))
        )
        llind = data.where(llcond).dropna(how='all').index
        if llind.size != 0:
            raise Exception(f"Records with indices {llind.values} contain "
                            f"latitude or longitude out of range. \n"
                            f"Change before proceeding!")

        return data

    def geoc_transform_loc(self,
                           data: pd.DataFrame,
                           ) -> pd.DataFrame:
        """ Transforms data locations to geocentric reference frame and stores
        corresponding transformations for the geodetic observables.

        Modifies data.
        """
        data['lon'] = data['lon'].where(data['lon'] <= 180, data['lon'] - 360)
        data['h'] = data['h'].where(data['h'].notna(), other=0)

        # indicate geodetic (0) or geocentric (1)
        data['geoc'] = data['geoc'].where(data['geoc'].notna(), other=0)
        data['geoc_colat'] = 90. - data['lat']
        data['geoc_rad'] = 6371.2 + data['h'] * 1e-3
        data['cd'] = 1.
        data['sd'] = 0.
        # apply geocentric correction
        temp, rad, cd, sd = latrad_in_geoc(
            np.radians(data['lat'].to_numpy()),
            data['h'].to_numpy().astype('float'))
        colat = 90 - np.degrees(temp)
        # only replace geodetic values
        gd_cond = data['geoc'] != 0
        data['geoc_colat'] = data['geoc_colat'].where(gd_cond, other=colat)
        data['geoc_rad'] = data['geoc_rad'].where(gd_cond, other=rad)
        data['cd'] = data['cd'].where(gd_cond, other=cd)
        data['sd'] = data['sd'].where(gd_cond, other=sd)

        return data

    def compile_data(self) -> None:
        """ Compiles data ready for quick use in geomagnetic field inversions

        Parameters
        ----------
        verbose
            if True, returns status report of number of data points

        Method updates data, loc_idx, loc, time, n_inp, idx_..., n_out,
        outputs, and errs
        """
        self.compiled = False
        # obtain index lists pointing to data quickly
        self.data.reset_index(inplace=True)
        # for quick access to stations
        uniq_loc, indices = np.unique(
            self.data[['geoc_colat', 'lon', 'geoc_rad',
                       'cd', 'sd']].to_numpy().astype('float'),
            return_inverse=True,
            axis=0,
        )
        # check geodetic/geocentric discrepancies
        uniq_loc_check = np.unique(
            self.data[['lat', 'lon', 'h']].to_numpy().astype('float'),
            axis=0,
        )
        if not (len(uniq_loc) == len(uniq_loc_check)):
            raise Exception('Location has both geocentric and geodetic data!')

        # DataFrame indices for D, I and F records
        self.n_inp = len(uniq_loc)
        self.idx_X = self.data.query('X==X').index
        self.idx_Y = self.data.query('Y==Y').index
        self.idx_Z = self.data.query('Z==Z').index
        self.idx_H = self.data.query('H==H').index
        self.idx_F = self.data.query('F==F').index
        self.idx_I = self.data.query('I==I').index
        self.idx_D = self.data.query('D==D').index
        self.idx_out = np.concatenate((self.idx_X, self.idx_Y, self.idx_Z,
                                       self.idx_H, self.idx_F, self.idx_I,
                                       self.idx_D)).flatten()
        # indices to quickly transform forward return matrices to the same
        # form as outputs
        self.idx_res = np.cumsum(
            [
                0,
                len(self.idx_X),
                len(self.idx_Y),
                len(self.idx_Z),
                len(self.idx_H),
                len(self.idx_F),
                len(self.idx_I),
                len(self.idx_D),
            ]
        )

        self.n_out = len(self.idx_out)
        # get same order as outputs
        self.loc = uniq_loc
        self.loc_idx = indices[self.idx_out]
        self.time = self.data['t'].loc[self.idx_out].to_numpy()
        # vector of data
        self.outputs = np.concatenate((self.data['X'].loc[self.idx_X],
                                       self.data['Y'].loc[self.idx_Y],
                                       self.data['Z'].loc[self.idx_Z],
                                       self.data['H'].loc[self.idx_H],
                                       self.data['F'].loc[self.idx_F],
                                       self.data['I'].loc[self.idx_I],
                                       self.data['D'].loc[self.idx_D])
                                      ).astype(float)
        # vector of errors
        self.std_out = np.concatenate((self.data['dX'].loc[self.idx_X],
                                       self.data['dY'].loc[self.idx_Y],
                                       self.data['dZ'].loc[self.idx_Z],
                                       self.data['dH'].loc[self.idx_H],
                                       self.data['dF'].loc[self.idx_F],
                                       self.data['dI'].loc[self.idx_I],
                                       self.data['dD'].loc[self.idx_D])
                                      ).astype(float)

    def __repr__(self):
        if self.n_out:
            return (
                f'Data from t={min(self.time)} to t={max(self.time)}\n'
                f'This dataset contains {self.n_out} records from '
                f'{self.n_inp} locations.\n'
                f'It consists of {self.idx_D.size} declinations, '
                f'{self.idx_I.size} inclinations and {self.idx_F.size} '
                f'intensities, {self.idx_X.size} x-data, '
                f'{self.idx_Y.size} y-data, {self.idx_Z.size} z-data, and '
                f'{self.idx_H.size} h-data.'
            )
        else:
            return 'This object does not contain any data.'


def read_geomagia(fname: Union[str, Path],
                  drop_duplicates: bool = True,
                  default_a95: float = 4.5,
                  **kw_args,
                  ) -> Optional[InputData]:
    """ Reads geomagia csv-file(s) format

    Parameters
    ----------
    fname
        string or Path-object of geomagia formatted file to read
    drop_duplicates
        if True, drops duplicate rows, i.e. rows that are exactly the same
    a95
        default error for alpha95, only used if no error is found in the file
    kw_args
        optional keyword argument(s) used for reading DataFrame with
        Input_data.read_data()

    Returns
    -------
        Pandas DataFrame of inputted geomagia data
    """
    # Missing values are indicated by either one of
    na = ('9999', '999', '999.9', 'nan', '-999', '-9999')
    # Read data as DataFrame
    try:
        dat = pd.read_csv(
            fname,
            usecols=['Age[yr.AD]', 'Sigma-ve[yr.]', 'Sigma+ve[yr.]',
                     'Ba[microT]', 'SigmaBa[microT]', 'Dec[deg.]',
                     'Inc[deg.]', 'Alpha95[deg.]', 'SiteLat[deg.]',
                     'SiteLon[deg.]'],
            na_values={'Sigma-ve[yr.]': (-1), 'Sigma+ve[yr.]': (-1),
                       'Ba[microT]': na, 'SigmaBa[microT]': na,
                       'Dec[deg.]': na, 'Inc[deg.]': na,
                       'Alpha95[deg.]': na},
            header=1, sep=',', skipinitialspace=True)
        # Rename columns
        ren_dict = {'Age[yr.AD]': 't', 'Sigma-ve[yr.]': 'dt_lo',
                    'Sigma+ve[yr.]': 'dt_up', 'Ba[microT]': 'F',
                    'SigmaBa[microT]': 'dF', 'Dec[deg.]': 'D',
                    'Inc[deg.]': 'I', 'SiteLat[deg.]': 'lat',
                    'SiteLon[deg.]': 'lon', 'Alpha95[deg.]': 'alpha95'}
    # TODO: make better statement to differentiate between sed and volc
    except ValueError:
        dat = pd.read_csv(
            fname,
            usecols=['Age[yr.BP]', 'Lat[deg.]', 'Lon[deg.]',
                     'Dec[deg.]', 'Inc[deg.]',
                     'SigmaDec[deg.]', 'SigmaInc[deg.]'],
            na_values={'Dec[deg.]': na, 'Inc[deg.]': na,
                        'SigmaDec[deg.]': na, 'SigmaInc[deg.]': na},
            header=1, sep=',', skipinitialspace=True, index_col=False)
        dat['Age[yr.BP]'] = -1 * dat['Age[yr.BP]'] + 1950
        # Rename columns
        ren_dict = {'Age[yr.BP]': 't', 'Dec[deg.]': 'D', 'Inc[deg.]': 'I',
                    'Lat[deg.]': 'lat', 'Lon[deg.]': 'lon',
                    'SigmaDec[deg.]': 'dD', 'SigmaInc[deg.]': 'dI'}
    dat.rename(ren_dict, inplace=True, axis='columns')

    # check if data already occurs and drop duplicates if wished so:
    if drop_duplicates:
        dat.drop_duplicates(subset=['lat', 'lon', 't'],
                            inplace=True)
    dat.dropna(subset=['lat', 'lon', 't'], inplace=True)
    dat.dropna(subset=['D', 'I', 'F'], how='all', inplace=True)
    dat.reset_index(inplace=True, drop=True)

    dat['alpha95'] = dat['alpha95'].where(
        (dat['D'].isna() & dat['I'].isna()) ^ dat['alpha95'].notna(),
        other=default_a95,
    )
    # dat['dat'] = np.clip(dat['dat'], 5, None)
    # Correct sqrt(2) error
    # dat['alpha95'] = np.clip(dat['alpha95'], np.sqrt(2) * 3.4, None)
    dat['dI'] = dat['alpha95'] * 57.3 / 140.
    cond = dat['D'].notna() & dat['I'].isna()  # fix alpha95 issues
    # Get the corresponding indices
    ind = dat.where(cond).dropna(how='all').index
    if ind.size != 0:
        warn(f"Records with indices {ind.values} contain "
             f"declination, but not inclination! No default error "
             f"(force_error_d) has been inputted.\n"
             f"To be able to use the provided data, these "
             f"records have been dropped from the output.",
             UserWarning)
        dat.drop(dat.where(cond).dropna(how='all').index,
                 inplace=True)

    dat['dD'] = dat['dI'] / np.abs(np.cos(np.deg2rad(dat['I'])))

    return dat
