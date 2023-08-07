import numpy as np


def latrad_in_geoc(lat: float,
                   h: float = 0.
                   ) -> [float, float, float, float]:
    """ Transforms the geodetic latitude to spherical coordinates
    Additionally, calculates the new radius and conversion factors
    source: Stevens et al. Aircraft control and simulation (2015)

    Parameters
    ----------
    lat
        geodetic latitude
    h
        height above geoid (m)

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
    # major axis earth (m)
    maj_axis = 6378137.0
    # minor axis earth (m)
    min_axis = 6356752
    # flattening
    flatt = (maj_axis - min_axis) / min_axis
    # eccentricity squared
    ecc2 = flatt * (2 - flatt)
    # prime vertical radius of curvature
    Rn = maj_axis / np.sqrt(1 - ecc2 * np.sin(lat)**2)
    # helpful parametrization
    n = ecc2 * Rn / (Rn + h)

    # calculate new geocentric latitude
    new_lat = np.arctan((1 - n) * np.tan(lat))
    # calculate new geocentric radius in km!
    new_rad = (Rn + h) * np.sqrt(1 - n * (2-n) * np.sin(lat)**2) * 1e-3
    # calculate changes in vector
    cd = np.cos(lat - new_lat)
    sd = np.sin(lat - new_lat)
    return new_lat, new_rad, cd, sd


def frechet_in_geoc(dx: np.ndarray,
                    dz: np.ndarray,
                    cd: np.ndarray,
                    sd: np.ndarray
                    ) -> [np.ndarray, np.ndarray]:
    """ Transforms geodetic frechet components to geocentric

    Parameters
    ----------
    dx
        polar component of the frechet matrix
    dz
        radial component of the frechet matrix
    cd
        cosine of latitude difference
    sd
        sine of latitude difference
    """
    dxnew = dx * cd[:, np.newaxis] + dz * sd[:, np.newaxis]
    dznew = dz * cd[:, np.newaxis] - dx * sd[:, np.newaxis]
    return dxnew, dznew
