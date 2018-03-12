"""
Helper library containing functions that perform coordinate conversion from
one format to another. For example, conversion from GPS coordinates in decimals
to degrees, minutes, seconds and hemisphere.

.. versionadded:: 0.1
.. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
"""

# Copyright (C) 2016-2017 Skylark Drones

from collections import namedtuple
from fractions import Fraction
from math import radians

import numpy as np


def deg_to_dms(lat, lng):
    """
    Convert coordinates from decimals to degrees, minutes, seconds and
    hemisphere

    :param lat: GPS Latitude in decimals
    :type lat: str
    :param lng: GPS Longitude in decimals
    :type lng: str

    :returns: GPS Latitude and Longitude in degrees, minutes, seconds and
        hemisphere
    :rtype: dict

    Example: ::

        {
            'Lat': {'Deg': 35, 'Min': 23, 'Sec': 45.555, 'Hem': 'N'},
            'Lng': {'Deg': 23, 'Min': 31, 'Sec': 55.555, 'Hem': 'E'}
        }
    """
    lat_hemisphere = "S" if lat[:1] == "-" else "N"
    lng_hemisphere = "W" if lng[:1] == "-" else "E"

    lat = abs(float(lat))
    lng = abs(float(lng))

    lat_deg = int(lat)
    lng_deg = int(lng)

    temp_lat_min = (lat - lat_deg) * 60
    lat_min = int(temp_lat_min)
    lat_sec = (temp_lat_min - lat_min) * 60

    temp_lng_min = (lng - lng_deg) * 60
    lng_min = int(temp_lng_min)
    lng_sec = (temp_lng_min - lng_min) * 60

    return {
        'Lat': {'Deg': lat_deg, 'Min': lat_min, 'Sec': lat_sec, 'Hem': lat_hemisphere},
        'Lng': {'Deg': lng_deg, 'Min': lng_min, 'Sec': lng_sec, 'Hem': lng_hemisphere}
    }


def dms_fractions_to_deg(gps_dms):
    """
    Convert coordinates in degrees, minutes, and seconds (fractions dict) into
    latitude and longitude

    The input gps_dms_fraction needs to be in the following format,

    Example: ::

        {
            'Lat': {'Deg': (14, 1), 'Min': (30, 1), 'Sec': (20574, 625), 'Hem': 'N'},
            'Lng': {'Deg': (78, 1), 'Min': (34, 1), 'Sec': (67971, 2500), 'Hem': 'E'}
        }

    :param gps_dms: GPS coordinates in degrees, minutes and seconds
    :type gps_dms: fractions dict

    :returns: Latitude and Longitude
    :rtype: tuple
    """
    DMS = namedtuple('DMS', ['deg', 'min', 'sec', 'hem'])
    FRAC = namedtuple('Time', ['num', 'den'])

    lat = DMS._make([FRAC._make(gps_dms['Lat'][key])
                     if key != 'Hem' else gps_dms['Lat']['Hem']
                     for key in ('Deg', 'Min', 'Sec', 'Hem')])

    lon = DMS._make([FRAC._make(gps_dms['Lng'][key])
                     if key != 'Hem' else gps_dms['Lng']['Hem']
                     for key in ('Deg', 'Min', 'Sec', 'Hem')])

    degree = float(lon.deg.num) / float(lon.deg.den)
    minute = float(lon.min.num) / float(lon.min.den) / 60
    second = float(lon.sec.num) / float(lon.sec.den) / 3600

    longitude_deg = degree + minute + second

    if gps_dms['Lng']['Hem'] == 'W':
        longitude_deg *= -1

    degree = float(lat.deg.num) / float(lat.deg.den)
    minute = float(lat.min.num) / float(lat.min.den) / 60
    second = float(lat.sec.num) / float(lat.sec.den) / 3600

    latitude_deg = degree + minute + second

    if gps_dms['Lat']['Hem'] == 'S':
        latitude_deg *= -1

    return latitude_deg, longitude_deg


def dms_decimals_to_fractions(coordinates):
    """
    Convert GPS degrees, minutes and seconds into fractions

    GPS coordinates inputted in degrees, minutes and seconds (dms) is converted
    into fraction. This is a requirement for writing exif tags to an image
    as specified in the official exif documentation at
    http://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf

    :param coordinates: GPS Coordinates
    :type coordinates: dict

    :returns: GPS coordinates in fractions
    :rtype: dict
    """
    DMS = namedtuple('DMS', ['deg', 'min', 'sec'])

    lat = DMS(Fraction(coordinates['Lat']['Deg']).limit_denominator(1),
              Fraction(coordinates['Lat']['Min']).limit_denominator(1),
              Fraction(coordinates['Lat']['Sec']).limit_denominator(1000000))

    lng = DMS(Fraction(coordinates['Lng']['Deg']).limit_denominator(1),
              Fraction(coordinates['Lng']['Min']).limit_denominator(1),
              Fraction(coordinates['Lng']['Sec']).limit_denominator(1000000))

    return {
        'Lat': {
            'Deg': (lat.deg.numerator, lat.deg.denominator),
            'Min': (lat.min.numerator, lat.min.denominator),
            'Sec': (lat.sec.numerator, lat.sec.denominator),
            'Hem': coordinates['Lat']['Hem']
        },

        'Lng': {
            'Deg': (lng.deg.numerator, lng.deg.denominator),
            'Min': (lng.min.numerator, lng.min.denominator),
            'Sec': (lng.sec.numerator, lng.sec.denominator),
            'Hem': coordinates['Lng']['Hem']
        }
    }


def lla_to_ecef(lat, lng, alt):
    """
    .. codeauthor:: Nihal Mohan <nihal@skylarkdrones.com>

    Convert latitude, longitude and altitude in decimal degrees to ECEF XYZ
    position in the WGS 84 Datum see
    http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    for source.

    :param float lat: latitude in decimal degrees
    :param float lng: longitude in decimal degrees
    :param float alt: altitude in meters

    :returns: XYZ(ECEF) in meters
    :rtype: tuple
    """

    # Convert to radians, cause they are better than degrees.
    lat = radians(lat)
    lng = radians(lng)

    # numpy is required for precision
    rad = np.float64(6378137.0)  # Radius of the Earth (in meters)
    f = np.float64(1.0 / 298.257223563)  # Flattening factor WGS84 Model

    # Intermediate constants for use in equation
    FF = (1.0 - f) ** 2
    geolat = np.arctan(FF * np.tan(lat))
    surface_radius = rad / np.sqrt(1 + (1 / FF - 1) * np.sin(geolat) ** 2)

    ecef_x = surface_radius * np.cos(geolat) * np.cos(lng) + \
             alt * np.cos(lat) * np.cos(lng)
    ecef_y = surface_radius * np.cos(geolat) * np.sin(lng) +\
             alt * np.cos(lat) * np.sin(lng)
    ecef_z = surface_radius * np.sin(geolat) + alt * np.sin(lat)

    return ecef_x, ecef_y, ecef_z
