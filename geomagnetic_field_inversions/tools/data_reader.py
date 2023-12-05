import numpy as np
from pathlib import Path
import os
import pandas as pd
from typing import Union
from ..data_prep import StationData


def read_write_korte(path: Path,
                     datafile: str,
                     outputfile: str,
                     ) -> None:
    """ Reads Korte et al. -like data and converts it

    Parameters
    ----------
    path
        Path to your file with data (and to be created output file)
    datafile
        Name of your data file
    outputfile
        Optional name of output file
    """
    newtime = 1
    file = open(path / datafile, 'r')
    f = open(path / outputfile, 'w')
    for r, row in enumerate(file):
        if r > 0:
            if r == newtime:
                time, a, locs = [float(x) for x in row.split()]
                newtime += int(locs) + 1
            else:
                inputs = np.array([float(x) for x in row.split()])
                nr_data = int(inputs[4])
                for i in range(nr_data):
                    # check for declination or inclination decidegrees
                    if inputs[7 + 3 * i] == 6 or inputs[7 + 3 * i] == 7:
                        inputs[5 + 3 * i] = inputs[5 + 3 * i] / 10
                        inputs[6 + 3 * i] = inputs[6 + 3 * i] / 10
                    # time, lat, lon, height, data, error, datatype
                    f.write(f'{inputs[0]}, {inputs[1]}, {inputs[2]}, {inputs[3]}, {inputs[5 + 3 * i]}, {inputs[6 + 3 * i]}, {inputs[7 + 3 * i]} \n')
    f.close()


def read_kortedata(path: Path,
                   datafile: str,
                   ) -> (np.ndarray, np.ndarray):
    """ Prepares data using the format of Korte et al.

    Parameters
    ----------
    path
        Path to your file with data (and to be created output file)
    datafile
        Name of your data file

    Returns
    -------
    stat_list
        list containing the different stations. Ready to be used for the
        geomagnetic field inversion
    """
    outputfile = 'OUTPUTFILE.txt'
    read_write_korte(path, datafile, outputfile)

    data = np.loadtxt(path / outputfile, delimiter=',')
    # lat, long, height (m) above geoid
    unique_coords, inverse = np.unique(data[:, 1:4], axis=0,
                                       return_inverse=True)  # sort on station

    red_data = data[:, [0, 4, 5, 6]]
    # per row one station: list of lists per type: time, data, error, type
    stations = [None] * len(unique_coords)
    for i in range(len(unique_coords)):
        station_data = red_data[np.where(inverse == i)]
        types, tinverse = np.unique(station_data[:, 3],
                                    return_inverse=True)  # sort on data type
        data_per_type = [None] * len(types)
        for t in range(len(types)):
            data_type = station_data[np.where(tinverse == t)]
            data_per_type[t] = data_type[
                data_type[:, 0].argsort()]  # order through time
        stations[i] = data_per_type
    # remove file
    try:
        os.remove(path / outputfile)
    except FileNotFoundError:
        pass

    # create stationlist and add data to class
    typedict = [None, 'x', 'y', 'z', 'hor', 'int', 'inc', 'dec']
    stat_list = []
    for i, station in enumerate(stations):
        name = f'Station {i}'  # we give the stations a name
        # initiate class with latitude, longitude, height, and name
        stat = StationData(lat=unique_coords[i, 0], lon=unique_coords[i, 1],
                           height=unique_coords[i, 2], name=name)
        # add types one by one
        for types in station:
            stat.add_data(typedict[int(types[0, 3])], data=types[:, :2].T,
                          time_factor=1, error=types[:, 2].flatten())
        stat_list.append(stat)
    return stat_list


_timecolumn = {'Age[yr.AD]': (lambda t: t),
               'Age[yr.BP]': (lambda t: -1 * t + 1950),
               'Age[yr.CE]': (lambda t: t),
               'Age[yr.BCE]': (lambda t: -1 * t)}
_deccolumn = {'DecAdj[deg.]': (lambda d: d), 'Dec[deg.]': (lambda d: d)}
_sdeccolumn = {'SigmaDec[deg.]': (lambda sd: sd)}
_inccolumn = {'IncAdj[deg.]': (lambda c: c), 'Inc[deg.]': (lambda c: c)}
_sinccolumn = {'SigmaInc[deg.]': (lambda sc: sc)}
_intcolumn = {'Ba[microT]': (lambda iy: iy)}
_sintcolumn = {'SigmaBa[microT]': (lambda siy: siy)}


def read_geomagia(path: Path,
                  datafile: str,
                  time_factor: int = 1,
                  input_error: np.ndarray = np.array([1, 1, 100]),
                  read_time: Union[dict, np.ndarray] = _timecolumn,
                  read_dec: Union[dict, np.ndarray] = _deccolumn,
                  read_sdec: Union[dict, np.ndarray] = _sdeccolumn,
                  read_inc: Union[dict, np.ndarray] = _inccolumn,
                  read_sinc: Union[dict, np.ndarray] = _sinccolumn,
                  read_int: Union[dict, np.ndarray] = _intcolumn,
                  read_sint: Union[dict, np.ndarray] = _sintcolumn,
                  ) -> list:
    """ Reads processed sedimentary data in geomagia format

    Parameters
    ----------
    path
        Path to file
    datafile
        name of file
    time_factor
        time factor of the time array, defaults to 1. If timearray is given in
        kiloyear tf should be 1000
    input_error
        Error of declination, inclination, and intensity when not provided
        by file
    read_time
        Either dictionary indicating which columns to read for time array
         + which function to apply to data read. We adopt CE terminology
        or np.ndarray containing relevant time in the right shape: len(data)
        If nothing passed in, default options are searched.
         When match is found search stops!
    read_dec, read_inc, read_int
        Either dictionary indicating which columns to read for relevant data
         + which function to apply to data read
        or np.ndarray containing relevant data in the right shape: len(data)
        If nothing passed in, default options are searched.
        If no match, no data of that type is added
         When match is found search stops!
    read_sdec, read_sinc, read_sint
        Simular to read_dec, read_inc, read_int,
         but then for standard deviations

    Returns
    -------
    stat_list
        list containing the different stations. Ready to be used for the
        geomagnetic field inversion
    """
    itemlist = [[read_dec, read_sdec],
                [read_inc, read_sinc],
                [read_int, read_sint]]
    datatypes = ['dec', 'inc', 'int']
    # read csv file
    data = pd.read_csv(path / datafile, skiprows=1, index_col=False,
                       low_memory=False)
    # read latitude and longitude and time
    try:  # sediment data
        nr = len(data['Lat[deg.]'].to_numpy())
        data_array = np.zeros((nr, 9))
        data_array[:, 0] = data['Lat[deg.]'].to_numpy()
        data_array[:, 1] = data['Lon[deg.]'].to_numpy()
    except KeyError:  # archeomagnetic/lava data
        nr = len(data['SiteLat[deg.]'].to_numpy())
        data_array = np.zeros((nr, 9))
        data_array[:, 0] = data['SiteLat[deg.]'].to_numpy()
        data_array[:, 1] = data['SiteLon[deg.]'].to_numpy()

    temp_time = read_col(data, read_time)
    # check time
    if temp_time is None:
        raise Exception('Incorrect columnname: time not read')
    else:
        data_array[:, 2] = temp_time

    # loop to scan through all possible datatypes using function
    for nr, read_item in enumerate(itemlist):
        data_temp = read_col(data, read_item[0])
        if data_temp is not None:
            data_array[:, 3 + 2 * nr] = data_temp
            sdata = read_col(data, read_item[1])
            if sdata is None:
                data_array[:, 4 + 2 * nr] = input_error[nr]
            else:
                data_array[:, 4 + 2 * nr] = sdata

    # sort per latitude/longitude pair
    slatlon, ind = np.unique(data_array[:, :2], return_inverse=True, axis=0)
    stat_list = []
    # loop through locations
    for i, ll in enumerate(slatlon):
        rows = np.argwhere(ind == i).flatten()
        station = StationData(ll[0], ll[1], name=f'Station_{i+1}')
        errorcount = 0
        for j, dtype in enumerate(datatypes):
            if j == 2:
                print(ll, data_array[rows][:, [2, 3+2*j, 4+2*j]])
            # first put data in array
            dat = data_array[rows][:, [2, 3+2*j, 4+2*j]]
            # find nonsense input data (equal to either -999 or +999)
            arg = np.argwhere(abs(dat[:, 1]) < 998).flatten()
            # check if not only zeros
            if np.any(dat[:, 1]) and np.any(arg):
                # find nonsense sigmas
                err = np.where(abs(dat[:, 2]) < 998, dat[:, 2], input_error[j])
                # add data to station
                station.add_data(dtype, dat[arg, :2].T,
                                 time_factor=time_factor, error=err[arg])
            else:
                print(f'No {dtype}-data added for Station_{i+1}')
                errorcount += 1
        # if station contains no data, do not add to list
        if errorcount < 3:
            stat_list.append(station)
        else:
            print(f"Station_{i+1} not added to list")
    return stat_list


def read_col(data, item):
    """ Reads data form csv file using dictionary or returns np.ndarray"""
    if type(item) is dict:
        for option in item:
            if option in data:
                return item[option](data[option].to_numpy())
    elif type(item) is np.ndarray:
        return item
