"""
.. versionadded:: 0.1
.. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>

**External Dependencies**

.. hlist::
   :columns: 2

   - piexif
   - `simplekml <https://pypi.python.org/pypi/simplekml/1.3.0>`_
   - git+https://github.com/smarnach/pyexiftool.git#egg=pyexiftool
   - exiftool binary

**Internal Dependencies**

.. hlist::
   :columns: 5

   - cameramodels.json
   - :class:`~quark.exceptions`
   - :class:`~quark.helpers.coords`
   - :class:`~quark.helpers.fileconstants`
   - :class:`~quark.helpers.utils`
"""

# Copyright (C) 2016-2017 Skylark Drones
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from shutil import copy2

import cv2
import exiftool
import numpy as np
import piexif
import simplekml
from piexif import InvalidImageDataError

from exceptions import NoExifFoundError, LeptonError
from helpers.coords import (dms_decimals_to_fractions, deg_to_dms,
                                  dms_fractions_to_deg)
from helpers.fileconstants import FileConstants

logger = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(__file__), 'cameramodels.json')) as camera_model_file:
    CAMERA_MODELS = json.load(camera_model_file)


class Image:
    """
    Image class holds an image and its related properties (name, path,
    creation time, lat, lng). The related properties are automatically read
    once the class is initialised through the constructor or when the absolute
    path of the image file is set. This will help simplify the business logic
    in other classes were images are dealt with.

    .. note::
        Only JPG images are supported due to constraints imposed by underlying
        libraries used.

    Another nice feature is built-in support for images compressed using
    Lepton. When a compressed image is passed as input, it will automatically
    be decompressed on the fly (by default to the same directory as the
    source image) and its attributes available for use.

    :param str image_abs_path: Absolute path of image (.jpg, .lep)
    :param float lat: Image latitude (Optional)
    :param float lng: Image longitude (Optional)
    :param float abs_alt: Image absolute altitude (Optional)
    :param bool is_geotagged: Flag to denote if input image is geotagged or
        not (Defaults to True)
    :param str decompress_to: Absolute path of folder to decompress image to
        (Optional)
    :raise LeptonError: If an error occurs during compressiong/decompression
        of image using Lepton
    :raise PermissionError: If write permission is not available to save
        image in the destination directory
    :raise ValueError: If input image is not in JPG or LEP format
    
    Example usage ::
    
        img = Image('/home/krnekhelesh/DSC001.JPG')
        print('{}, {}, {}'.format(img.name, img.path, img.creation_time))

        comp_img = Image('/home/krnekhelesh/DSC001.lep'
                         decompress_to='/home/krnekhelesh/Decompressed-Images')
        print('{}, {}, {}'.format(comp_img.name, comp_img.lat, comp_img.lng))
    """
    lepton_executable = os.path.join(
        FileConstants().QUARK_LIB_DIR,
        'execs',
        'lepton' if sys.platform == 'linux' else 'lepton.exe'
    )

    # This startup flag is required to ensure that the lepton command prompt
    # does not appear when compressing an image using a GUI program like DMC.
    startup_info = None
    if sys.platform == 'win32':
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    def __init__(self, image_abs_path, lat=None, lng=None, abs_alt=None,
                 is_geotagged=True, decompress_to=None):
        self._abs_path = None
        self._base_path = None
        self._image_name = None
        self._comp_img_path = None

        self._creation_time = None
        self._brightness = None
        self._lat = lat
        self._lng = lng
        self._abs_alt = abs_alt
        self._pitch = None
        self._yaw = None
        self._roll = None
        self._relative_alt = None
        self._focal_length = None
        self._sensor_width = None
        self._sensor_height = None
        self._exif_dict = None
        self._hsv_brightness = None
        self._blur = None
        self._image_for_opencv = None
        self._camera_model = None

        self._is_geotagged = is_geotagged

        self._set_full_path(image_abs_path, decompress_to)


    def __repr__(self):
        return "{}({}, {}, {}, {}, {})".format(self.__class__.__name__,
                                               self._image_name,
                                               self._creation_time,
                                               self._lat, self._lng,
                                               self._abs_alt)

    def __eq__(self, other):
        return self._abs_path == other.full_path

    def _set_full_path(self, abs_path, decompress_to=None):
        if not os.path.isfile(abs_path) or abs_path is None:
            raise ValueError('Input image path is not valid!')

        if not abs_path.lower().endswith(('.jpg', '.lep')):
            raise InvalidImageDataError('Only JPG images are supported!')

        if abs_path.lower().endswith('.lep'):
            self._comp_img_path = abs_path
            abs_path = self._decompress_lep_image(abs_path, decompress_to)

        self._abs_path = abs_path
        self._base_path, self._image_name = os.path.split(self._abs_path)
        self._exif_dict = piexif.load(self._abs_path)
        self._read_creation_time_exif()
        self._read_brightness_exif()
        self._read_focal_length_exif()
        self._get_sensor_dimensions()
        if self._is_geotagged:
            self._read_gps_exif()
            self._read_alt_exif()

        if self._exif_dict['Exif'] == {}:
            logger.error('No EXIF metadata found in {image}'.format(image=abs_path))
            raise NoExifFoundError

    @property
    def full_path(self):
        """
        Absolute path of image **(READ ONLY)**
        """
        return self._abs_path

    @property
    def compressed_path(self):
        """
        Absolute path of the compressed image file **(READ ONLY)**
        """
        return self._comp_img_path

    @property
    def hsv_brightness(self):
        """
        Image brightness **(READ ONLY)**
        """
        if self._hsv_brightness is None:
            self._get_hsv_brightness()
        return self._hsv_brightness

    @property
    def blur(self):
        """
        Image Blurriness **(READ ONLY)**
        """
        if self._blur is None:
            self._get_blur()
        return self._blur

    @property
    def name(self):
        """
        Image name **(READ ONLY)**
        """
        return self._image_name

    @property
    def path(self):
        """
        Base path of image **(READ ONLY)**
        """
        return self._base_path

    @property
    def creation_time(self):
        """
        Creation time of image read from EXIF metadata 
        """
        return self._creation_time

    @creation_time.setter
    def creation_time(self, value):
        if not isinstance(value, datetime):
            err_msg = 'Input creation time is not a DateTime object'
            logger.error(err_msg)
            raise TypeError(err_msg)
        self._creation_time = value

    @property
    def brightness(self):
        """
        Brightness of an image read from EXIF metadata **(READ ONLY)**        
        """
        return self._brightness

    @property
    def lat(self):
        """
        GPS latitude of image in decimals **(READ ONLY)**
        """
        return self._lat

    @property
    def lng(self):
        """
        GPS longitude of image in decimals **(READ ONLY)**
        """
        return self._lng

    @property
    def abs_alt(self):
        """
        GPS altitude of image **(READ ONLY)**        
        """
        return self._abs_alt

    @property
    def camera_model(self):
        """
        Camera model name with which image is captured
        """
        return self._camera_model

    @property
    def roll(self):
        """
        Flight roll angle when the image was taken 
        """
        return self._roll

    @roll.setter
    def roll(self, value):
        try:
            self._roll = float(value)
        except (TypeError, ValueError) as error:
            raise Exception('Invalid roll value!') from error

    @property
    def pitch(self):
        """
        Flight pitch angle when the image was taken 
        """
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        try:
            self._pitch = float(value)
        except (TypeError, ValueError) as error:
            raise Exception('Invalid pitch value!') from error

    @property
    def yaw(self):
        """
        Flight yaw angle when the image was taken 
        """
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        try:
            self._yaw = float(value)
        except (TypeError, ValueError) as error:
            raise Exception('Invalid yaw value!') from error

    @property
    def relative_alt(self):
        """
        Flight relative altitude
        """
        return self._relative_alt

    @relative_alt.setter
    def relative_alt(self, value):
        try:
            self._relative_alt = float(value)
        except (TypeError, ValueError) as error:
            raise Exception('Invalid relative altitude value!') from error

    @property
    def focal_length(self):
        """
        Focal length of the camera when the image was taken **(READ ONLY)**
        """
        return self._focal_length

    @property
    def sensor_width(self):
        """
        Sensor width of the camera obtained from a database **(READ ONLY)**
        """
        return self._sensor_width

    @property
    def sensor_height(self):
        """
        Sensor height of the camera obtained from a database **(READ ONLY)**
        """
        return self._sensor_height

    def set_coords(self, lat, lng, abs_alt):
        """
        Set image coordinates
        
        :param float lat: Latitude in decimals 
        :param float lng: Longitude in decimals
        :param float abs_alt: Absolute altitude in float
        """
        self._lat = float(lat)
        self._lng = float(lng)
        self._abs_alt = float(abs_alt)

    def set_orientation(self, roll, pitch, yaw):
        """
        Set image orientation
        
        :param float roll: Roll in degrees
        :param float pitch: Pitch in degrees
        :param float yaw: Yaw in degrees
        """
        self._roll = float(roll)
        self._pitch = float(pitch)
        self._yaw = float(yaw)

    def write_gps_exif(self, copy_to=None):
        """
        Write gps exif data into an image

        :param str copy_to: Absolute path of file to write exif into

        :returns: True or False to indicate success of write operation
        :rtype: bool
        """
        try:
            coordinate = dms_decimals_to_fractions(deg_to_dms(str(self._lat),
                                                              str(self._lng)))
            altitude = Fraction(self._abs_alt).limit_denominator(100)
            altitude = (altitude.numerator, altitude.denominator)

            exif_dict = piexif.load(self._abs_path)
            gps_ifd = {
                piexif.GPSIFD.GPSLatitudeRef: coordinate['Lat']['Hem'],
                piexif.GPSIFD.GPSLatitude: (
                    (coordinate['Lat']['Deg']),
                    (coordinate['Lat']['Min']),
                    (coordinate['Lat']['Sec'])
                ),
                piexif.GPSIFD.GPSLongitude: (
                    (coordinate['Lng']['Deg']),
                    (coordinate['Lng']['Min']),
                    (coordinate['Lng']['Sec'])
                ),
                piexif.GPSIFD.GPSLongitudeRef: coordinate['Lng']['Hem'],
                piexif.GPSIFD.GPSAltitude: altitude
            }
            exif_dict['GPS'] = gps_ifd
            exif_bytes = piexif.dump(exif_dict)
            if copy_to is None:
                piexif.insert(exif_bytes, self._abs_path)
            else:
                copy2(self._abs_path, copy_to)
                piexif.insert(exif_bytes, os.path.join(copy_to, self._image_name))
        except ValueError:
            return False
        else:
            return True

    def read_xmp(self):
        """
        Read image's XMP data which at the moment includes the gimbal pitch,
        roll and yaw values using exiftool.
        """
        exiftool.executable = os.path.join(
            FileConstants().QUARK_LIB_DIR,
            'execs',
            'exiftool' if sys.platform == 'linux' else 'exiftool.exe'
        )
        exif_tool = exiftool.ExifTool()
        exif_tool.start()

        flight_state = exif_tool.get_tags(['GimbalPitchDegree',
                                           'GimbalRollDegree',
                                           'GimbalYawDegree',
                                           'RelativeAltitude'],
                                          self._abs_path)
        print(flight_state)

        _pitch = flight_state.get('XMP:GimbalPitchDegree', None)
        _roll = flight_state.get('XMP:GimbalRollDegree', None)
        _yaw = flight_state.get('XMP:GimbalYawDegree', None)
        _relative_altitude = flight_state.get('XMP:RelativeAltitude', None)
        print('Relative alt image : {}'.format(_relative_altitude))

        self._pitch = float(_pitch) if _pitch is not None else None
        self._roll = float(_roll) if _roll is not None else None
        self._yaw = float(_yaw) if _yaw is not None else None
        self._relative_alt = float(_relative_altitude) if _relative_altitude is not None else None
        print(self._relative_alt)
        exif_tool.terminate()

    def _read_creation_time_exif(self):
        """
        Read image creation time from image EXIF tags
        """
        try:
            creation_time_str = self._exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized].decode()
            self._creation_time = datetime.strptime(creation_time_str, '%Y:%m:%d %H:%M:%S')
        except (KeyError, ValueError):
            # In the event that an invalid image file is provided or creation
            # time metadata is not available, return None
            self._creation_time = None

    def _read_brightness_exif(self):
        """
        Read image brightness value from image EXIF tags
        """
        try:
            brightness = self._exif_dict['Exif'][piexif.ExifIFD.BrightnessValue]
            self._brightness = brightness[0] / brightness[1]
        except (KeyError, ValueError):
            # In the event of KeyError due to missing Exif data, return None
            # In the event that an invalid image file is provided, return None
            self._brightness = None

    def _read_focal_length_exif(self):
        """
        Read Focal length from image EXIF data
        """
        try:
            focal_length = self._exif_dict['Exif'][piexif.ExifIFD.FocalLength]
            self._focal_length = focal_length[0] / focal_length[1]
        except (KeyError, ValueError, ZeroDivisionError):
            # In the event of KeyError due to missing Exif data, set None
            # In the event that an invalid image file is provided, set None
            self._focal_length = None

    def _get_sensor_dimensions(self):
        """
        Get sensor dimensions for camera from database

        .. codeauthor:: Samarth Hattangady <samarth@skylarkdrones.com>
        """
        try:
            # DJI stores the camera model as a null padded byte string :
            # b'DJI\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0'
            # Thus we have to clean it up a little before use
            camera_model = ' '.join([str(self._exif_dict['0th'][piexif.ImageIFD.Make], 'utf-8').split('\x00')[0],
                                     str(self._exif_dict['0th'][piexif.ImageIFD.Model], 'utf-8').split('\x00')[0]])
            self._camera_model = camera_model
            self._sensor_width = CAMERA_MODELS[camera_model]['SensorWidth']
            self._sensor_height = CAMERA_MODELS[camera_model]['SensorHeight']
        except (KeyError, ValueError):
            # In case of KeyError with missing Exif data/sensor data or
            # invalid image, leave sensor dimensions as None
            pass

    def _read_gps_exif(self):
        """
        Read GPS coordinates from image EXIF metadata
        """
        try:
            lng_deg, lng_min, lng_sec = self._exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
            lng_hem = self._exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode()
            lat_deg, lat_min, lat_sec = self._exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
            lat_hem = self._exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode()

            gps_dms = {
                'Lat': {'Deg': lat_deg, 'Min': lat_min, 'Sec': lat_sec, 'Hem': lat_hem},
                'Lng': {'Deg': lng_deg, 'Min': lng_min, 'Sec': lng_sec, 'Hem': lng_hem}
            }
            self._lat, self._lng = dms_fractions_to_deg(gps_dms)
        except (KeyError, ValueError, TypeError):
            self._lat, self._lng = None, None

    def _get_blur(self):
        """
        The reason variance of Laplacian method works  to calculate
        bluriness is due to the definition of the Laplacian operator
        itself, which is used to measure the 2nd derivative of an image.
        The Laplacian highlights regions of an image containing rapid
        intensity changes, much like the Sobel and Scharr operators
        (kernels). And, just like these operators, the Laplacian is often
        used for edge detection. The assumption here is that if an image
        contains high variance then there is a wide spread of responses,
        both edge-like and non-edge like, representative of a normal,
        in-focus image. But if there is very low variance, then there is
        a tiny spread of responses, indicating there are very little
        edges in the image. As we know, the more an image is blurred, the
        less edges there are.
        """
        image = cv2.imread(self._abs_path)
        self._blur = cv2.Laplacian(image, cv2.CV_64F).var()

    def _get_hsv_brightness(self):
        """
        Standard approach.
        The V ("value") attribute in the HSV format representation of an image
        represents the brightness of the pixel. Now simply take the average of
        brightness of all pixels, and we can imply this as the image brightness
        """
        image = cv2.imread(self._abs_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        self._hsv_brightness = np.mean(np.mean(v, axis=1), axis=0)

    def _read_alt_exif(self):
        """
        Read GPS altitude from image EXIF metadata
        """
        try:
            abs_alt = self._exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
            self._abs_alt = abs_alt[0] / abs_alt[1]
        except (KeyError, ValueError, TypeError):
            self._abs_alt = None

    # Inclusion for white object detecton and opencv filters starts here

    def apply_bilateral_filter(self, diameter, sigma_color, sigma_space,
                               **kwargs):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This function returns bilateral filtered image. This filter filters out
        noises in color preserving the edges between different colors. For more
        information, check OpenCV bilateral filter implementation
        (cv2.bilateralFilter_).

        :param int diameter: Diameter of each pixel neighborhood that is used
            during filtering. If it is non-positive, it is computed from
            sigmaSpace.
        :param double sigma_color: Filter sigma in the color space. A larger
                value of the parameter means that farther colors within the
                pixel neighborhood (see sigmaSpace) will be mixed together,
                resulting in larger areas of semi-equal color.
        :param double sigma_space: Filter sigma in the coordinate space. A
                larger value of the parameter means that farther pixels will
                influence each other as long as their colors are close enough
                (see sigma_color). When d>0 , it specifies the neighborhood size
                regardless of sigma_space. Otherwise, d is proportional to
                sigma_space.
        :param str ColorSpace: Grayscale/BGR. Default is BGR. (optional)
        :param double BorderType: Default is cv2.BORDER_DEFAULT. (optional)

        :return: filtered image either in grayscale or BGR depending on color
            space

        .. image:: ../_images/bilateral.jpg
            :align: center
        .. _cv2.bilateralFilter: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
        """

        if self._image_for_opencv is None:
            self._image_for_opencv = cv2.imread(self._abs_path, cv2.IMREAD_COLOR)
        image = self._image_for_opencv
        if kwargs.get('ColorSpace', 'BGR') == 'Grayscale':
            image = cv2.cvtColor(self._image_for_opencv, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(image,
                                   diameter,
                                   sigma_color,
                                   sigma_space,
                                   kwargs.get('BorderType', cv2.BORDER_DEFAULT))

    def apply_adaptive_equalizer(self, **kwargs):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This function is an implementation of clahe(contrast limited adaptive
        histogram equalized) object from OpenCV, which enhances image quality
        by improving brightness but preserving contrast caused due to
        perspective in objects appearing in image. In an image in some portion
        where shadowing due to poor lighting condition occurs, this filter
        corrects it while preserving brightness condition for good lighting
        condition portion depending on proper ClipLimit and TileSize. For more
        information about clahe, please check OpenCV clahe implementation
        (cv2.createClahe_).

        :param double ClipLimit: Threshold for contrast limiting. Default is
            40.0. (optional)
        :param Size TileSize: Size of grid for histogram equalization. Default
            is (8, 8). Input image will be divided into equally sized
            rectangular tiles. TileSize defines the number of tiles in row and
            column. (optional)
        :param str ColorSpace: Color space where clahe should be applied.
            Default is "BGR". (optional)

        :return: adaptive equalized image

        .. image:: ../_images/clahe.jpg
            :align: center
            :scale: 60 %

        .. _cv2.createClahe: https://docs.opencv.org/3.0-beta/modules/cudaimgproc/doc/histogram.html
        """

        if self._image_for_opencv is None:
            self._image_for_opencv = cv2.imread(self._abs_path, cv2.IMREAD_COLOR)
        image = self._image_for_opencv

        clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 40.0),
                                tileGridSize=kwargs.get('clip_limit', (8, 8)))

        if kwargs.get('ColorSpace', 'BGR') == 'Grayscale':
            image = cv2.cvtColor(self._image_for_opencv, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
        else:
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image[:, :, 1] = clahe.apply(image[:, :, 1])
            image[:, :, 2] = clahe.apply(image[:, :, 2])
        return image

    def apply_dilate_erode(self, kernel_size, **kwargs):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This function takes kernel_size as input argument and returns
        eroded/dilated/both operated(first dilated and then eroded) image
        depending on dilate_iteration and erode_iteration is defined as input
        argument or not. This filter basically removes point noises.

        :param tuple kernel_size: tuple of size 2. This decides kernel size for
            the operation.
        :param int dilate_iteration: Number of iteration for dilation
            operation. If not defined dilation operation is avoided. (optional)
        :param int erode_iteration: Number of iteration for erosion
            operation. If not defined erosion operation is avoided. (optional)

        :return: adaptive equalized image

        .. image:: ../_images/dilation_n_erosion.jpg
           :align: center
        """
        if self._image_for_opencv is None:
            self._image_for_opencv = cv2.imread(self._abs_path, cv2.IMREAD_COLOR)
        image = self._image_for_opencv
        kernel = np.ones(kernel_size, np.uint8)
        if kwargs.get('dilate_iteration', False):
            image = cv2.dilate(image,
                               kernel,
                               iterations=kwargs['dilate_iteration'])
        if kwargs.get('erode_iteration', False):
            image = cv2.erode(image,
                              kernel,
                              iterations=kwargs['erode_iteration'])
        return image

    @staticmethod
    def _get_last_bin_of_histogram(single_channel_image, hist_method):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This method takes single channel image data like grayscale image or any
        channel of BGR data as input and returns value of the last color bin
        (basically white color) from the histogram calculation method given by
        hist_method. In some images it is seen that white color code deviates
        from its absolute value 255 to lesser value due to poor image quality.
        Therefore, to get exact color code value for a white color appearing in
        0-255 scale, we have to know the last color bin of histogram
        effectively. This method gives the correct information of color code of
        a color in an image, given that white color object is there in image.
        Other parameters of numpy.histogram is neglected here because this
        methods primary concern is to get last bin value or color code of an
        image.(check _numpy.histogram for better understanding of numpy
        histogram calculation)
        :param single_channel_image: Grayscale/single channel of an image
        :param str/int/scalars hist_method: Histogram calculation method used
            in numpy.
        :return: Exact color code of a color defined in 0-255 scale.
        :raise ValueError: Raises ValueError if the more than one channel image
            is passed as argument.

        .. _numpy.histogram: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
        """
        if len(single_channel_image.shape) != 2:
            raise ValueError("Image should be single channel image data.")
        else:
            arr = np.reshape(single_channel_image,
                             (1, np.product(single_channel_image.shape)))
            hist_bins, bin_edges = np.histogram(arr[0], hist_method)
        return bin_edges[len(bin_edges) - 1]

    def _get_last_bin_of_hist_for_3channel(self,
                                           three_channel_image,
                                           hist_method):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This method takes BGR or RGB image data as input and returns value of
        the color bin that corresponds to white color approximately in three
        channels from the histogram calculation method given by hist_method.
        :param three_channel_image: BGR image
        :param str/int/scalars hist_method: Histogram calculation method used
            in numpy.
        :return: Exact color code of a color defined in 0-255 scale.
        :raise ValueError: Raises ValueError if other than three channel image
            is passed as argument.
        """
        if three_channel_image.shape[2] != 3:
            raise ValueError("BGR image data should contain three channel data")
        try:
            last_bin_0 = self._get_last_bin_of_histogram(
                three_channel_image[:, :, 0],
                hist_method)
            last_bin_1 = self._get_last_bin_of_histogram(
                three_channel_image[:, :, 1],
                hist_method)
            last_bin_2 = self._get_last_bin_of_histogram(
                three_channel_image[:, :, 2],
                hist_method)
        except ValueError:
            raise ValueError
        return np.uint8([last_bin_0, last_bin_1, last_bin_2])

    def apply_color_threshold(self, **kwargs):
        """
        .. codeauthor:: Dweepjyoti Malakar <dweepmalakar@skylarkdrones.com>
        .. versionadded:: Quark-0.2

        This function takes image threshold parameter as input argument and
        returns binary image. This function is important to extract some defined
        colored object in an image.

        :param str ColorSpace: BGR/Grayscale. Default is Grayscale, because it
            is easier to threshold in Grayscale being single channel
            representative for each coclor band.
        :param dict LowerLimit: It is a dict to determine the lower limit for
            threshold the colored image to binary image. It can have following
            keys.

                * *HistogramMethod* (``str``):
                    It can have numpy histogram calculation methods_. It is same
                    for BGR/Grayscale images.
                * *ColorBound* (``double/list of doubles``):
                    It should be defined if HistogramMethod is used to determine
                    the lower limit. It is a one double value for thresholding
                    in Grayscale color space or a list of 3 doubles for
                    thresholding in BGR color space. This parameter is defined
                    such that it comprises of differences of the color in which
                    the image should be thresholded from absolute white in 0-255
                    scale (8 bit encoded image scale). That means if in a normal
                    good image lower pink is defined as [255, 204, 255]. Then to
                    identify lower pink value in bad conditioned image, we have
                    to negate this value from absolute white, ie [255, 255, 255]
                    That means [0, 51, 0] as ColorBound parameter. (optional but
                    mandatory if HistogramMethod is defined in LowerLimit dict)
                * *NormalTresholdValue* (``double/list of double``):
                    This is the normal color value encoding in Grayscale(one
                    double value) or BGR(a list of 3 doubles). Suppose normal
                    lower pink is in BGR color space by [255, 153, 255], then
                    this value should be defined as [255, 153, 255]. (optional)
        :param dict UpperLimit: It is a dict to determine the upper limit for
            threshold the colored image to binary image. It can have following
            keys.

                * *HistogramMethod* (``str``):
                    It can have numpy histogram calculation methods_. It is same
                    for BGR/Grayscale images.
                * *ColorBound* (``double/list of double``):
                    It should be defined if HistogramMethod is used to determine
                    the upper limit. It is a one double value for thresholding
                    in Grayscale color space or a list of 3 doubles for
                    thresholding in BGR color space. This parameter is defined
                    such that it comprises of differences of the color in which
                    the image should be thresholded from absolute white in 0-255
                    scale (8 bit encoded image scale). That means if in a normal
                    good image upper pink is defined as [204, 0, 102]. Then to
                    identify pink value in bad conditioned image, we have to
                    negate this value from absolute white, ie [255, 255, 255].
                    That means [51, 255, 153] as ColorBound parameter. (optional
                    but mandatory if HistogramMethod is defined in LowerLimit
                    dict)
                * *NormalTresholdValue* (``double/list of double``):
                    This is the normal color value encoding in Grayscale(one
                    double value) or BGR(a list of 3 doubles). Suppose normal
                    upper pink is in BGR color space by [204, 0, 102], then this
                    value should be defined as [204, 0, 102]. (optional)
        :return: Binary image depending on color limits given by user. Color
            defined  will come as white and other portion will become black.

        .. note::
            If nothing is defined as LowerLimit or UpperLimit dict, then for BGR
            images [120.0, 120.0, 120.0] is defined as lower limit and [255.0,
            255.0, 255.0] is defined as upper limit and for Grayscale image
            thresholding it will take 120.0 as lower limit and 255.0 as upper
            limit.

        .. _methods: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
        """
        if self._image_for_opencv is None:
            self._image_for_opencv = cv2.imread(self._abs_path, cv2.IMREAD_COLOR)
        image = self._image_for_opencv

        if kwargs.get('ColorSpace', 'Grayscale') == "BGR":
            if 'LowerLimit' in kwargs and 'HistogramMethod' in kwargs[
                'LowerLimit'] and 'ColorBound' in kwargs['LowerLimit']:
                hist_method = kwargs['LowerLimit']['HistogramMethod']
                if len(kwargs['LowerLimit']['ColorBound']) != 3:
                    raise ValueError('ColorBound should be a list of 3 doubles')
                color_bound = kwargs['LowerLimit']['ColorBound']
                lower_limit = self._get_last_bin_of_hist_for_3channel(
                    hist_method) - np.uint8(color_bound)

            elif 'LowerLimit' in kwargs and 'NormalTresholdValue' in kwargs[
                'LowerLimit']:
                lower_limit = np.uint8(
                    kwargs['LowerLimit']['NormalTresholdValue'])
            else:
                lower_limit = np.uint8([120, 120, 120])

            if 'UpperLimit' in kwargs and 'HistogramMethod' in kwargs[
                'UpperLimit'] and 'ColorBound' in kwargs['UpperLimit']:
                hist_method = kwargs['UpperLimit']['HistogramMethod']
                if len(kwargs['UpperLimit']['ColorBound']) != 3:
                    raise ValueError('ColorBound should be a list of 3 doubles')
                color_bound = kwargs['UpperLimit']['ColorBound']
                upper_limit = self._get_last_bin_of_hist_for_3channel(
                    hist_method) - np.uint8(color_bound)

            elif 'UpperLimit' in kwargs and 'NormalTresholdValue' in kwargs[
                'UpperLimit']:
                upper_limit = np.uint8(
                    kwargs['UpperLimit']['NormalTresholdValue'])
            else:
                upper_limit = np.uint8([255.0, 255.0, 255.0])

            thresh = cv2.inRange(image, lower_limit, upper_limit)

            return thresh

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if 'LowerLimit' in kwargs and 'HistogramMethod' in kwargs[
                'LowerLimit'] and 'ColorBound' in kwargs['LowerLimit']:
                hist_method = kwargs['LowerLimit']['HistogramMethod']
                if len(kwargs['LowerLimit']['ColorBound']) != 1:
                    raise ValueError('ColorBound should be a scalar value')
                color_bound = kwargs['LowerLimit']['ColorBound']
                lower_limit = self._get_last_bin_of_histogram(
                    hist_method) - np.uint8(color_bound)

            elif 'LowerLimit' in kwargs and 'NormalTresholdValue' in kwargs[
                'LowerLimit']:
                lower_limit = np.uint8(
                    kwargs['LowerLimit']['NormalTresholdValue'])
            else:
                lower_limit = np.uint8(120.0)

            if 'UpperLimit' in kwargs and 'HistogramMethod' in kwargs[
                'UpperLimit'] and 'ColorBound' in kwargs['UpperLimit']:
                hist_method = kwargs['UpperLimit']['HistogramMethod']
                if len(kwargs['UpperLimit']['ColorBound']) != 1:
                    raise ValueError('ColorBound should be a list of 3 doubles')
                color_bound = kwargs['UpperLimit']['ColorBound']
                upper_limit = self._get_last_bin_of_histogram(
                    hist_method) - np.uint8(color_bound)

            elif 'UpperLimit' in kwargs and 'NormalTresholdValue' in kwargs[
                'UpperLimit']:
                upper_limit = np.uint8(
                    kwargs['UpperLimit']['NormalTresholdValue'])
            else:
                upper_limit = np.uint8(255.0)

        ret, thresh = cv2.threshold(image, lower_limit, upper_limit,
                                    cv2.THRESH_BINARY)

        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def compress_losslessly(self, comp_folder_path=None):
        """
        Lossless compression of an image using Lepton which is a compression
        library created by Dropbox. It achieves 22% compression on an average.

        .. note::
            You will need to decompress the compressed image using Lepton or
            using this class again to be able to view it in an image viewer
            or use it else where.

        :param str comp_folder_path: Absolute path of folder to store
            compressed image (Optional)
        :raise ValueError: If input image is not in JPG format or if the
            compression output folder does not exist
        :raise PermissionError: If comp_folder_path directory is read-only
        :raise LeptonError: If an error occurs while lepton is trying to
            compress the image.
        """
        if not self._image_name.lower().endswith('.jpg'):
            raise ValueError('Lepton algorithm only accepts JPG images!')

        comp_img_name = self._image_name[:-4] + '.lep'
        if comp_folder_path is None:
            comp_img = os.path.join(self._base_path, comp_img_name)
        elif not os.path.isdir(comp_folder_path):
            raise ValueError('Compression output folder cannot be found!')
        else:
            comp_img = os.path.join(comp_folder_path, comp_img_name)

        if not os.access(os.path.dirname(comp_img), os.W_OK):
            raise PermissionError('Write permission not granted to compress '
                                  'image to {}'.format(os.path.dirname(comp_img)))

        arg = [self.lepton_executable, self._abs_path, comp_img]

        try:
            subprocess.call(arg,
                            stdout=open(os.devnull, 'wb'),
                            stderr=subprocess.STDOUT,
                            startupinfo=self.startup_info)
        except subprocess.CalledProcessError as process_error:
            logger.exception('Lossless compression using Lepton failed due'
                             'to a process error!')
            raise LeptonError from process_error
        else:
            self._comp_img_path = comp_img

    def _decompress_lep_image(self, comp_img_path, decomp_img_folder=None):
        """
        Decompress image compressed using lossless compression utility Lepton.
        By default, the decompressed image is stored with the same name in the
        same directory as the source image. This behavior can be overwritten
        by providing a output folder path.

        :param str comp_img_path: Absolute path of compressed image
        :param str decomp_img_folder: Absolute path of folder to store
            decompressed image (Optional)
        :raise ValueError: If decompression output folder does not exist
        :raise PermissionError: If decomp_img_folder is read-only
        :raise LeptonError: If an error occurs while Lepton is trying to
            decompress the image.
        """
        decomp_img_name = os.path.basename(comp_img_path)[:-4] + '.JPG'
        if decomp_img_folder is None:
            decomp_img = os.path.join(os.path.dirname(comp_img_path),
                                      decomp_img_name)
        elif not os.path.isdir(decomp_img_folder):
            raise ValueError('Decompression folder cannot be found!')
        else:
            decomp_img = os.path.join(decomp_img_folder, decomp_img_name)

        if not os.access(os.path.dirname(decomp_img), os.W_OK):
            raise PermissionError('Write permission not granted to decompress '
                                  'image to {}'.format(os.path.dirname(decomp_img)))

        arg = [self.lepton_executable, comp_img_path, decomp_img]
        try:
            subprocess.call(arg,
                            stdout=open(os.devnull, 'wb'),
                            stderr=subprocess.STDOUT,
                            startupinfo=self.startup_info)
        except subprocess.CalledProcessError as process_error:
            logger.exception('Lossless decompression using Lepton failed due'
                             'to a process error!')
            raise LeptonError from process_error
        else:
            return decomp_img


class Images(list):
    """
    Images class is a subclass of list. It holds all the images in a given
    folder sorted by creation time. Invalid images and those with no creation
    time are ignored.

    .. note::
        Each item in the list is of type :class:`~quark.helpers.image.Image`.

    Example usage ::

        images = Images().load_folder('/path/to/image_folder')
        for img in images:
            print('{}, {}, {}'.format(img.name, img.path, img.creation_time))

        images.generate_kmz()
    """

    def load_folder(self, folder_path, decompress_folder_path=None):
        """
        Parse through folder and create list containing only images

        :param str folder_path: Absolute path of images folder
        :param str decompress_folder_path: Absolute path of folder where
            decompressed images should be created (if input image is a
            compressed image)
        """
        if not os.path.isdir(folder_path):
            err_msg = 'Input folder path {} is not a valid folder or ' \
                      'cannot be found!'.format(folder_path)
            logger.error(err_msg)
            raise ValueError(err_msg)

        for img in os.listdir(folder_path):
            if not img.lower().endswith(('.jpg', '.lep')):
                continue

            try:
                image = Image(os.path.join(folder_path, img),
                              decompress_to=decompress_folder_path)
                if image.creation_time is None:
                    logger.warning('Creation time of {} not '
                                   'available!'.format(image.name))
                else:
                    self.append(image)
            except NoExifFoundError:
                pass
            except InvalidImageDataError:
                logger.warning('EXIF data is either corrupt or written by '
                               'non-compliant software in {}'.format(img))
                pass

        self.sort(key=lambda x: x.creation_time)
        return self

    def _generate_photo_location(self, save_folder_path=None,
                                 output_format='kmz'):
        """
        Geographically represent the images data set by plotting all the
        images as points on a map (kml/kmz). This is then viewable using
        Google Earth.

        :param str save_folder_path: By default, resulting kml/kmz file is
            stored in the same directory as the image data set (Optional)
        :param str output_format: Output format, accepted values are kmz/kml.
            Defaults to kmz format.
        """

        if not len(self):
            raise ValueError('Input images data set is empty!')

        geotagged_images = [image
                            for image in self
                            if not all((image.lng is None, image.lat is None))]

        if not len(geotagged_images):
            raise ValueError('The entire input data set of images are '
                             'not geotagged!')

        output_file_name = FileConstants().PHOTO_LOCATION_FILE_NAME[:-3] \
                           + output_format.lower()
        if save_folder_path is None:
            output_file = os.path.join(self[0].path, output_file_name)
        elif not os.path.isdir(save_folder_path):
            raise ValueError('Save folder path provided cannot be found!')
        else:
            output_file = os.path.join(save_folder_path, output_file_name)

        kml = simplekml.Kml()
        for image in geotagged_images:
            kml_point = kml.newpoint()
            kml_point.description = image.name
            kml_point.coords = [(image.lng, image.lat, image.abs_alt)]
            if output_format == 'kmz':
                kml_point.style.iconstyle.icon.href = 'files/yellow-circle-dot.png'
                kml_point.style.iconstyle.scale = 0.4

        if output_format == 'kmz':
            kml.addfile(os.path.join(
                FileConstants().QUARK_LIB_DIR,
                'images',
                'yellow-circle-dot.png'
            ))

        # If kmz file exists, remove it as it throws a FileExistsError on
        # Windows when trying to overwrite the file
        if os.path.exists(output_file):
            os.remove(output_file)

        kml.savekmz(output_file) if output_format == 'kmz' else kml.save(output_file)

    def generate_kmz(self, save_folder_path=None):
        """
        .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
        .. versionadded:: Quark-0.1.1

        Geographically represent the images data set by plotting all the
        images as points on a map (kmz file) which can be viewed on
        Google Earth.

        .. warning ::
            When integrating Quark into your application, be sure to add the
            images assets folder to your cx_freeze setup file. Failing which
            cx_freeze will skip the assets folder leading to them being
            unavailable in production mode.

        :param str save_folder_path: By default, resulting kmz file is
            stored in the same directory as the image data set (Optional)
        :raise ValueError: If images data set is either empty or if not
            even a single image is georeferenced or if the folder path
            provided does not exist.
        """
        self._generate_photo_location(save_folder_path)

    def generate_kml(self, save_folder_path=None):
        """
        .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
        .. versionadded:: Quark-0.1.1

        Geographically represent the images data set by plotting all the
        images as points on a map (kml file) which can be viewed on
        Google Earth.

        :param str save_folder_path: By default, resulting kml file is
            stored in the same directory as the image dataset (Optional)
        :raise ValueError: If images data set is either empty or if not
            even a single image is georeferenced or if the folder path
            provided does not exist.
        """
        self._generate_photo_location(save_folder_path, output_format='kml')

    def generate_csv(self, save_folder_path=None):
        """
        .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
        .. versionadded:: Quark-0.1.1

        Generate a comma separated file (CSV) of the images data set

        :param str save_folder_path: By default, resulting csv file is created
            in the same directory as the image dataset (Optional)
        :raise ValueError: If images data set is either empty or if not
            even a single image is georeferenced or if the folder path
            provided does not exist.
        """

        if not len(self):
            raise ValueError('Input images data set is empty')

        geotagged_images = [
            image
            for image in self
            if not all((image.lng is None, image.lat is None))
        ]

        if not len(geotagged_images):
            raise ValueError('The entire input data set of images are '
                             'not geotagged!')

        output_file_name = FileConstants().PHOTO_DETAIL_FILE_NAME
        if save_folder_path is None:
            output_file = os.path.join(self[0].path, output_file_name)
        elif not os.path.isdir(save_folder_path):
            raise ValueError('Save folder path provided cannot be found!')
        else:
            output_file = os.path.join(save_folder_path, output_file_name)

        # If a CSV file exists with the same name, remove it as it throws
        # a FileExistsError on Windows when trying to overwrite the file.
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, 'w') as photo_details_file:
            photo_details_file.write(
                'Name, Lat, Lng, Abs_Alt, Roll, Pitch, Yaw\n'
            )
            for image in geotagged_images:
                image.read_xmp()
                photo_details_file.write(
                    '{name}, {lat}, {lng}, {abs_alt}, {roll}, {pitch}, '
                    '{yaw}\n'.format(name=image.name,
                                     lat=image.lat,
                                     lng=image.lng,
                                     abs_alt=image.lng,
                                     roll=image.roll,
                                     pitch=image.pitch,
                                     yaw=image.yaw)
                )

    def geotag_images(self, save_folder_path=None):
        """
        .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
        .. versionadded:: Quark-0.1.1

        Geotag the images data set by embedding GPS information like
        latitude, longitude and altitude into the EXIF metadata of an image.

        .. note::
            Before using this function, ensure that the images in the list
            have GPS information stored in their attributes. You can use
            :py:func:`~quark.helpers.image.Image.set_coords` to do this.
            This function merely writes/embeds that information into the
            EXIF metadata.

        :param str save_folder_path: By default, the original data set is
            modified. If a folder path is provided, then the data set is
            first duplicated to the folder path and then geotagged.
        :raise ValueError: If images data set is either empty or if not
            even a single image has GPS information that is used to embed
            it into the EXIF metadata or if the folder path provided does
            not exist.
        """
        if not len(self):
            raise ValueError('Input images data set is empty')

        images_with_gps_metadata = [
            image
            for image in self
            if not all((image.lng is None, image.lat is None))
        ]

        if not len(images_with_gps_metadata):
            raise ValueError('The entire input data set of images do '
                             'any contain georeference data!')

        if save_folder_path is None:
            # Defaults to embedding GSP information into the original
            # image itself.
            output_folder = None
        elif not os.path.isdir(save_folder_path):
            raise ValueError('Save folder path provided cannot be found!')
        else:
            output_folder = os.path.join(
                save_folder_path, FileConstants().GEOTAG_IMG_FOLDER_NAME
            )

            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)

        for image in images_with_gps_metadata:
            image.write_gps_exif(copy_to=output_folder)
