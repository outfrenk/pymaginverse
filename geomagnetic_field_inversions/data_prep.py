import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from typing import Union, Literal, Optional
import warnings

_IpMethods = Literal['polyfit', 'USpline']
_DataTypes = Literal['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']


def interpolate_data(t: Union[np.ndarray, list],
                     y: Union[np.ndarray, list],
                     order: int,
                     smoothing: Optional[float] = None,
                     method: _IpMethods = 'polyfit'):
    """
    Interpolates data over its total timerange.
    Required for sampling purposes in the inversion

    Parameters
    ----------
    t
        timevector of corresponding datavector y
    y
        datavector containing inclination, declination, intensity,
        Mx, My, Mz, or horizontal magnetic component
    order
        order of fitting
    smoothing
        degree of smoothing
    method
        method of fitting data. either NumPy's polyfit or
        SciPy's UnivariateSpline
    """

    if method == 'polyfit':
        polyn = np.polyfit(t, y, order)  # order of polynome
        linefit = np.poly1d(polyn)
        return linefit
    elif method == 'USpline':
        unispline = UnivariateSpline(t, y, k=order, s=smoothing)
        return unispline
    else:
        raise Exception('Method %s not recognized' % method)


def check_data(ttype: _DataTypes,
               data: Union[list, np.ndarray],
               time_factor: int,
               error: Union[list, np.ndarray] = None):
    """
    Checks if the data is correctly inputted and datatypes exist.
    Raises an exception if inc/dec data does not fit expected range.
    The function sorts the data in time from old to new.

    Parameters
    ----------
    ttype
        string containing the datatype of the corresponding data array
    data
        numpy array containing 2 rows: data and time
    time_factor
            unit of the time vector. E.g. if timevector is inputted in kyr,
            factor should be 1.000. If timevector is inputted in Myr, factor
            should be 1.000.000.
    error
        error of the data, which should be list or numpy array with length
        equal length to time vector. If no error is included in data,
        a default error is added: 1 deg. inc/dec error and 100 muT else.

    Returns data and error array merged together and sorted by time.
    """
    datatypes = ['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']
    # default error values: muT and radians
    errordict = {"x": 100, "y": 100, "z": 100, "hor": 100,
                 "int": 100, "inc": np.radians(1), "dec": np.radians(1)}
    # Initiate arrays for storing starting and ending time data
    if len(data) != 2:
        raise Exception('Data has incorrect format (%i). Did you include time,'
                        ' and magnetic data?' % len(data))
    if ttype in datatypes:
        if ttype == 'inc':
            maxd, mind = max(data[1]), min(data[1])
            if maxd > 90 or mind < -90:
                warnings.warn('Incl. falls outside (-90, 90) range.')
        if ttype == 'dec':
            maxd, mind = max(data[1]), min(data[1])
            if maxd > 180 or mind < -180:
                warnings.warn('Decl. falls outside (-180, 180) range')
    else:
        raise ValueError('Type %s not recognized' % ttype)

    # assign and check error
    if error is None:
        error = np.full(len(data[0]), errordict[ttype])
    elif len(error) == 1:
        error = np.full(len(data[0]), error)
    elif len(error) == len(data[0]):
        pass
    else:
        raise Exception(f'error has incorrect length: {len(error)},\n'
                        f'It should have length {len(data[0])} or length 1')
        
    data = np.array(data)
    data[0] *= time_factor
    sorting = data[0].argsort()  # sort on time; small to large
    data = data[:, sorting]  # data, time
    error = error[sorting]
    return np.vstack((data, error)).tolist()


class StationData:
    """
    Class designed to contain and fit all data per station. It stores
    time (kyr) with data (x, y, z, hor, inc, dec, int) assuming either
    microteslas or degrees.
    """

    def __init__(self,
                 lat: float,
                 lon: float):
        """
        Set latitude and longitude of station
        """
        self.lat = lat
        self.lon = lon

        # initiate empty lists
        self.data = []
        self.types = []
        self.fit_data = None
        self.time_factor = 1

    def add_data(self,
                 ttype: _DataTypes,
                 data: Union[list, np.ndarray],
                 time_factor: int,
                 error: Union[list, np.ndarray] = None):
        """
        Add magnetic data and data type to the class
        Also performs a first order check on the data

        Parameters
        ----------
        ttype
            array containing the magnetic data types stored in data
            should be either: 'x', 'y', 'z', 'hor', 'inc' (degrees,
            'dec' (degrees), or 'int'
        data
            nested list that stores contains two rows: a time vector and
            magnetic data (x, y, z, hor, inc, dec, or int):
            [[time_1, time_2, time_3, ..., time_n]
             [inc_1, inc_2, inc_3, ..., inc_n]]
        time_factor
            unit of the time vector. E.g. if timevector is inputted in kyr,
            factor should be 1.000. If timevector is inputted in Myr, factor
            should be 1.000.000.
        error
            error of the data, which should be list or numpy array with length
            equal length of time vector. If no error is included in data,
            a default error is added: 1 deg. inc/dec error and 100 muT else.

        self.data contains time, data, and error
        """
        self.data.append(check_data(ttype, data, time_factor, error))
        self.types.append(ttype)

    def fitting(self,
                order: Union[int, list, np.ndarray] = None,
                smoothing: Union[int, float, list, np.ndarray] = None,
                method: _IpMethods = None,
                output: Optional[str] = None):
        """
        Fits a function to the data, required for running the inverse at
        values between datapoints as well.

        Parameters
        ----------
        order
            order of fit
        smoothing
            smoothing applied to fit
        method
            method of fitting; either polyfit or UnivariateSpline
        output
            if a path is passed to output, a datafigure is created

        Returns
        -------
        Optionally returns a figure of the fitted data
        """
        if order is None:
            order = np.full(len(self.types), 10)
        if smoothing is None:
            smoothing = np.full(len(self.types), None)
        if method is None:
            method = np.full(len(self.types), 'polyfit')

        self.fit_data = [None] * len(self.types)
        for i in range(len(self.types)):
            self.fit_data[i] = interpolate_data(self.data[i][0],
                                                self.data[i][1],
                                                order[i],
                                                smoothing[i],
                                                method[i])
            if output is not None:
                time_arr = np.linspace(self.data[i][0][0],
                                       self.data[i][0][-1], 100)
                fig, ax = plt.subplots()
                ax.set_title('Fitting %s of data' % self.types[i])
                if self.types[i] == 'inc' or self.types[i] == 'dec':
                    ax.set_ylabel('%s (degrees)' % self.types[i])
                else:
                    ax.set_ylabel('%s' % self.types[i])
                ax.set_xlabel('Time (kyr)')
                ax.plot(time_arr, self.fit_data[i](time_arr),
                        label='fit', color='orange')
                ax.scatter(self.data[i][0], self.data[i][1], label='data')
                ax.legend()
                plt.savefig('%s/plot_%s.png' % (output, self.types[i]))
