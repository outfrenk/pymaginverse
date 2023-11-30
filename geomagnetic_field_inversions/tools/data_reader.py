import numpy as np
from pathlib import Path
import os
import pandas as pd
from typing import Union
from ..data_prep import StationData

def read_write(path: Path,
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


def read_data(path: Path,
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
    unique_coords
        coordinate list of the different location.
        Sorted lat (degrees), lon (degrees), and height (m) above geoid
    stations
        per row (one location): list of lists per datatype through time:
        Each list (length time) contains time, data, error, type
    """
    outputfile = 'OUTPUTFILE.txt'
    read_write(path, datafile, outputfile)

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
    return unique_coords, stations


def read_geomagia(path: Path,
                  datafile: str,
                  read_int: Union[list, np.ndarray] = None,
                  time_factor: int = 1,
                  input_error: np.ndarray = np.array([1, 1, 100])
                  ) -> list:
    """ Reads processed sedimentary data in geomagia format

    Parameters
    ----------
    path
        Path to file
    datafile
        name of file
    read_int
        Either string indicating which columns to read for intensity data
        or np.ndarray containing the intensity data in the right shape
        (2, len(data)), first row contains data, second row contains errors.
        If None (default), no intensity data is added.
    time_factor
        time factor of the time array, defaults to 1. If timearray is given in
        kiloyear tf should be 1000
    input_error
        Error of declination, inclination, and intensity when not provided
        by file

    Returns
    -------
    stat_list
        list containing the different stations. Ready to be used for the
        geomagnetic field inversion
    """
    data = pd.read_csv(path / datafile, skiprows=1, index_col=False,
                       low_memory=False)
    nr = len(data['Lat[deg.]'].to_numpy())
    data_array = np.zeros((nr, 9))
    try:
        data_array[:, 0] = data['Age[yr.BP]'].to_numpy() * -1 + 1950
    except KeyError:
        data_array[:, 0] = data['Age[yr.AD]'].to_numpy()
    data_array[:, 1:7] = data[['Lat[deg.]', 'Lon[deg.]', 'Dec[deg.]',
                               'SigmaDec[deg.]', 'Inc[deg.]', 'SigmaInc[deg.]'
                               ]].to_numpy()
    print(data_array[:, 0])
    data_types = ['dec', 'inc']
    if type(read_int) is list and len(read_int) == 2:
        data_array[:, 7] = data[read_int[0]].to_numpy()
        data_array[:, 8] = data[read_int[1]].to_numpy()
        data_types.append('int')
    elif type(read_int) is np.ndarray:
        assert read_int.shape == (2, nr), 'incorrect shape intensity data,' \
                                          f'should be (2, {nr})'
        data_array[:, 7:] = read_int
        data_types.append('int')
    else:
        print('No intensity added')

    # sort per latitude/longitude pair
    slatlon, ind = np.unique(data_array[:, 1:3], return_inverse=True, axis=0)
    stat_list = []
    # loop through locations
    for i, ll in enumerate(slatlon):
        rows = np.argwhere(ind == i).flatten()
        station = StationData(ll[0], ll[1], name=f'Station_{i+1}')
        for j, dtype in enumerate(data_types):
            # first put data in array
            dat = data_array[rows][:, [0, 3+2*j, 4+2*j]]
            # find nonsense input data
            arg = np.argwhere(abs(dat[:, 1]) < 800).flatten()
            # find nonsense sigmas
            err = np.where(abs(dat[:, 2]) < 800, dat[:, 2], input_error[j])
            # add data to station
            station.add_data(dtype, dat[arg, :2].T, time_factor=time_factor,
                             error=err[arg])
        stat_list.append(station)

    return stat_list
