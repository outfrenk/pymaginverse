import numpy as np
from pathlib import Path


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
              outputfile: str = None,
              overwrite: bool = True,
              ) -> (np.ndarray, np.ndarray):
    """ Prepares data using the format of Korte et al.

    Parameters
    ----------
    path
        Path to your file with data (and to be created output file)
    datafile
        Name of your data file
    outputfile
        Optional name of output file
    overwrite
        Whether to overwrite existing output files. Defaults to True

    Returns
    -------
    unique_coords
        coordinate list of the different location.
        Sorted lat (degrees), lon (degrees), and height (m) above geoid
    stations
        per row (one location): list of lists per datatype through time:
        Each list (length time) contains time, data, error, type
    """
    if outputfile is None:
        outputfile = 'output'
    if overwrite or not (path / outputfile).is_file():
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
    return unique_coords, stations
