import numpy as np
import pandas as pd
from pathlib import Path
from warnings import warn
from typing import Union


class InputData(object):
    """Class that contains geomagnetic data and locations in order to perform
    a geomagnetic field inversion

    Attributes
    ----------
    data : DataFrame
        The DataFrame containing all data.
    n_inp : int
        The number of inputs, i.e. rows of the DataFrame
    inputs : array
        The inputs (colat, lon, radius, time)
    idx_X, idx_Y, idx_Z, idx_H, idx_D, idx_I, idx_F : index
        The indices of X, Y, Z, H, dec, inc, and int records in the DataFrame.
    n_out : int
        The number of outputs, i.e. the total number of individual data points.
    outputs : array
        The outputs (order is X, Y, Z, H, dec, inc, int)
    errs : array
        The output-errors.
    compiled : boolean
        Whether indices for data have been compiled
    """
    def __init__(self):
        self.data = pd.DataFrame(
            columns=['lat', 'lon', 'rad', 't', 'colat', 'X', 'dX', 'Y', 'dY',
                     'Z', 'dZ', 'H', 'dH', 'D', 'dD', 'I', 'dI', 'F', 'dF',
                     'alpha95', 'geoc'])
        self.inputs, self.n_inp = 0, 0
        self.idx_X,  self.idx_Y, self.idx_Z, self.idx_H = 0, 0, 0, 0
        self.idx_D,  self.idx_I, self.idx_F = 0, 0, 0
        self.n_out = 0
        self.outputs, self.errs = np.zeros(0), np.zeros(0)
        self.compiled = False

    def read_data(self,
                  dfs: Union[list, pd.DataFrame],
                  drop_duplicates: bool = False,
                  x_err=10, y_err=10, z_err=10,
                  h_err=10, f_err=10, a95=10,
                  force_error_d: float = None):
        """ Reads Pandas DataFrame(s) and stores in class

        Parameters
        ----------
        dfs
            (list with instances of) Pandas DataFrame. Contains data + error
             on either X/Y/Z/(H) component or inc/dec/int components or both.
            Header should be:
             'lat' (latitude), 'lon' (longitude), 'rad' (optional: radius),
             't' (time), 'X'/'Y'/'Z'/'H' (x/y/z/h), 'I'/'D'/'F' (inc/dec/int),
             'dX'/'dY'/'dZ'/'dH' (x/y/z/h-error),
             'dD'/'dI'/'dF' (dec/inc/int error),
             'alpha95' (a95 error for both dec and inc error)
             'geoc' (only set to 1 if data is obtained in geocentric dataframe)
        drop_duplicates
            if True, drop duplicates with same lat, lon, time, and radius
        x_err, y_err, z_err, h_err, f_err
            default errors (10) if no error is found in DataFrame
        a95
            default error for alpha95, only used if no error for
             dec or/and inc are found.
        force_error_d
            If None, removes data with dec data, but no dec error & inc data
            If float is set, will set dec error to that float

        Method updates data
        """
        default_error = [x_err, y_err, z_err, h_err, f_err]
        # check for list -> enables processing multiple DataFrames
        if not isinstance(dfs, list):
            dfs = [dfs]

        for df in dfs:
            # this evades errors due to missing column names
            df = df.reindex(df.columns.union(self.data.columns, sort=False),
                            axis=1)
            # drop rows not containing basic information
            df.dropna(subset=['lat', 'lon', 't'], how='any', inplace=True)
            # check latitude and longitude
            llcond = (abs(df['lat']) > 90) | (
                    (df['lon'] > 360) | (df['lon'] < -180))
            llind = df.where(llcond).dropna(how='all').index
            if llind.size != 0:
                raise Exception(f"Records with indices {llind.values} contain "
                                f"latitude or longitude out of range. \n"
                                f"Change before proceeding!")
            df['lon'].where(df['lon'] <= 180, df['lon'] - 360, inplace=True)
            df['colat'] = 90 - df['lat']
            df['rad'].where(df['rad'].notna(), other=6371.2, inplace=True)

            # change zero error values to default if no data
            for i, dstr in enumerate(['X', 'Y', 'Z', 'H', 'F']):
                df[f'd{dstr}'].where(
                    df[dstr].isna() ^ df[f'd{dstr}'].notna(),
                    other=default_error[i], inplace=True)
            df['alpha95'].where(
                (df['D'].isna() & df['I'].isna()) ^ df['alpha95'].notna() ^
                (df['dD'].notna() & df['dI'].notna()),
                other=a95, inplace=True)
            # df['dF'] = np.clip(df['dF'], 5, None)
            # Correct sqrt(2) error
            # df['alpha95'] = np.clip(df['alpha95'], np.sqrt(2) * 3.4, None)
            df['dI'].where(df['dI'].notna(),
                           other=df['alpha95'] * 57.3 / 140., inplace=True)

            cond = df['D'].notna() & df['I'].isna()  # fix alpha95 issues
            # Get the corresponding indices
            ind = df.where(cond).dropna(how='all').index
            if ind.size != 0:
                if force_error_d:
                    df['dD'].where(cond, other=force_error_d, inplace=True)
                else:
                    warn(f"Records with indices {ind.values} contain "
                         f"declination, but not inclination! No default error "
                         f"(force_error_d) has been inputted.\n"
                         f"To be able to use the provided data, these "
                         f"records have been dropped from the output.",
                         UserWarning)
                    df.drop(df.where(cond).dropna(how='all').index,
                            inplace=True)

            df['dD'].where(df['dD'].notna(),
                           other=df['alpha95'] * 57.3 / 140. / np.abs(
                               np.cos(np.deg2rad(df['I']))), inplace=True)

            # check if data already occurs and drop duplicates if wished so:
            if drop_duplicates:
                df.drop_duplicates(subset=['lat', 'lon', 'rad', 't'],
                                   inplace=True)
            # add dataframe to big dataframe
            self.data = pd.concat([self.data, df[self.data.columns]],
                                  ignore_index=True)
            self.compiled = False

    def compile_data(self,
                     drop_duplicates: bool = True,
                     verbose: bool = False):
        """ Compiles data ready for quick use in geomagnetic field inversions

        Parameters
        ----------
        drop_duplicates
            if True, drops duplicate rows, i.e. rows are exactly the same
        verbose
            if True, returns status report of number of data points

        Method updates data, inputs, n_inp, idx_..., n_out, outputs, and errs
        """
        # obtain index lists pointing to data quickly
        data = self.data
        # check for duplicates
        if drop_duplicates:
            data.drop_duplicates(inplace=True)
        data.reset_index(inplace=True)
        # for quick access
        self.inputs = np.asfortranarray(
            data[['colat', 'lon', 'rad', 't']].to_numpy().T)
        # DataFrame indices for D, I and F records
        self.n_inp = len(data)
        self.idx_X = data.query('X==X').index
        self.idx_Y = data.query('Y==Y').index
        self.idx_Z = data.query('Z==Z').index
        self.idx_H = data.query('H==H').index
        self.idx_D = data.query('D==D').index
        self.idx_I = data.query('I==I').index
        self.idx_F = data.query('F==F').index
        self.n_out = self.idx_X.size + self.idx_Y.size + self.idx_Z.size\
                     + self.idx_H.size + self.idx_D.size + self.idx_I.size\
                     + self.idx_F.size
        # vector of data
        self.outputs = np.concatenate((data['X'].loc[self.idx_X],
                                       data['Y'].loc[self.idx_Y],
                                       data['Z'].loc[self.idx_Z],
                                       data['H'].loc[self.idx_H],
                                       data['D'].loc[self.idx_D],
                                       data['I'].loc[self.idx_I],
                                       data['F'].loc[self.idx_F]))
        # vector of errors
        std_out = np.concatenate((data['X'].loc[self.idx_X],
                                  data['Y'].loc[self.idx_Y],
                                  data['Z'].loc[self.idx_Z],
                                  data['H'].loc[self.idx_H],
                                  data['D'].loc[self.idx_D],
                                  data['I'].loc[self.idx_I],
                                  data['F'].loc[self.idx_F]))
        self.errs = std_out**2
        self.compiled = True
        if verbose:
            print(f'Data from t={min(self.inputs[3])} to '
                  f't={max(self.inputs[3])}\n'
                  f'This dataset contains {self.n_out} records from '
                  f'{self.n_inp} locations.\n'
                  f'It consists of {self.idx_D.size} declinations, '
                  f'{self.idx_I.size} inclinations and {self.idx_F.size} '
                  f'intensities, {self.idx_X.size} x-data, '
                  f'{self.idx_Y.size} y-data, {self.idx_Z.size} z-data, and '
                  f'{self.idx_H.size} h-data.')


def read_geomagia(fnames: Union[list, str, Path],
                  id_attr: InputData = None,
                  return_df: bool = False,
                  **kw_args):
    if id_attr is None:
        id_attr = InputData()
    dats = pd.DataFrame(columns=id_attr.data.columns)

    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
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
        dats = pd.concat([dats, dat], ignore_index=True)

    dats.dropna(subset=['lat', 'lon', 't'], inplace=True)
    dats.dropna(subset=['D', 'I', 'F'], how='all', inplace=True)
    dats.reset_index(inplace=True)
    if return_df:
        return dats
    else:
        id_attr.read_data(dats, **kw_args)
        return id_attr
