import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from typing import Union, Literal, Optional, Callable

_IpMethods = Literal['polyfit', 'USpline', 'linear']
_DataTypes = Literal['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']


def interpolate_data(t: Union[np.ndarray, list],
                     y: Union[np.ndarray, list],
                     order: int = None,
                     smoothing: Optional[float] = None,
                     method: _IpMethods = 'linear'
                     ) -> Callable[[np.ndarray], np.ndarray]:
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
        method of fitting data. either NumPy's polyfit,
        SciPy's UnivariateSpline or interp1d (linear)

    Returns
    -------
    linefit, unispline, linear
        fitting function
    """
    if method == 'polyfit':
        polyn = np.polyfit(t, y, order)  # order of polynome
        linefit = np.poly1d(polyn)
        return linefit
    elif method == 'USpline':
        unispline = UnivariateSpline(t, y, k=order, s=smoothing)
        return unispline
    elif method == 'linear':
        linear = interp1d(t, y, kind='linear')
        return linear
    else:
        raise Exception('Method %s not recognized' % method)


def check_data(ttype: _DataTypes,
               data: Union[list, np.ndarray],
               time_factor: int,
               error: Union[list, np.ndarray]
               ) -> list:
    """
    Checks if the data is correctly inputted and datatypes exist.
    Raises an exception if inc/dec data does not fit expected range.
    The function sorts the data in time from old to new.

    Parameters
    ----------
    ttype
        string containing the datatype of the corresponding data array
    data
        numpy array containing 2 rows: time and data
    time_factor
            unit of the time vector. E.g. if timevector is inputted in kyr,
            factor should be 1.000. If timevector is inputted in Myr, factor
            should be 1.000.000.
    error
        error of the data, which should be float, list, or numpy array with
        length equal length to time vector.

    Returns
    -------
        data and error array merged together and sorted by time.
    """
    datatypes = ['x', 'y', 'z', 'hor', 'inc', 'dec', 'int']
    # Initiate arrays for storing starting and ending time data
    if len(data) != 2:
        raise Exception('Data has incorrect format (%i). Did you include time,'
                        ' and magnetic data?' % len(data))
    if ttype not in datatypes:
        raise ValueError('Type %s not recognized' % ttype)

    # assign and check error
    try:
        if len(error) == 1:
            error = np.full(len(data[0]), error[0])
        elif len(error) == len(data[0]):
            pass
        else:
            raise Exception(f'error has incorrect length: {len(error)},\n'
                            f'It should have length 1 or {len(data[0])}!')
    except TypeError:
        raise TypeError(f'error is of incorrect type: {type(error)},\n'
                        f'It should be of type: list of np.ndarray')
        
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
                 lon: float,
                 height: float = 0,
                 name: Optional[str] = None):
        """
        Set latitude & longitude in degrees.
        optionally geodetic height in metres and optionally name of station
        """
        if lon > 180:  # if longitude inputted between 0 and 360
            lon -= 360
            print('Longitude changed from %.2f to %.2f' % (lon, lon+360))
        if lat > 90 or lat < -90 or lon > 180 or lon < -180:
            raise ValueError(f'Your latitude is {lat} and longitude is {lon};'
                             ' latitude should be between -90 and 90 degrees,'
                             ' longitude between -180 and 180 degrees.')
        self.lat = lat
        self.lon = lon
        self.height = height
        if name is not None:
            self.__name__ = name
        else:
            self.__name__ = 'station'

        # initiate empty lists
        self.data = []
        self.types = []
        self.fit_data = None
        self.time_factor = 1

    def add_data(self,
                 ttype: _DataTypes,
                 data: Union[list, np.ndarray],
                 time_factor: int,
                 error: Union[list, np.ndarray]
                 ) -> None:
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
            equal to length of time vector. Either nT or degrees

        Creates or modifies
        -------------------
        self.data (creates)
            contains time, data, and error
        self.types (creates)
            corresponding datatype (see _DataTypes)
        """
        self.data.append(check_data(ttype, data, time_factor, error))
        self.types.append(ttype)
        self.fit_data = [None] * len(self.types)

    def fitting(self,
                ttype: str = None,
                order: float = None,
                smoothing: float = None,
                method: _IpMethods = None
                ) -> Optional[plt.Axes]:
        """
        Fits a function to the data, required for running the inverse at
        values between datapoints as well. Default running this function will
        result into a linear fit for all datatypes. Otherwise, fit per datatype

        Parameters
        ----------
        ttype
            which datatype to fit. if omitted, all datatype will be fitted
            linearly
        order
            order of fit
        smoothing
            smoothing applied to fit
        method
            method of fitting; either linear, polyfit, or UnivariateSpline

        Creates or modifies
        -------------------
        self.fit_data
            function that fits and interpolates the data
        """
        if ttype is None:
            for i in range(len(self.types)):
                self.fit_data[i] = interpolate_data(self.data[i][0],
                                                    self.data[i][1],
                                                    method='linear')
        else:
            row = np.where(np.char.find(self.types, ttype) == 0)[0]
            if len(row) == 0:
                raise Exception('datatype not found, try again')
            self.fit_data[row[0]] = interpolate_data(
                self.data[row[0]][0], self.data[row[0]][1],
                order, smoothing, method)
