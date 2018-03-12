import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class WhiteObjectDetector:
    def __init__(self):
        self.white_object_detection_parameter_file = os.path.join(
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            'config',
            'whiteobjectdetection.json')
        if not os.path.exists(self.white_object_detection_parameter_file):
            print("White object detection parameter file does not exists")
        with open(self.white_object_detection_parameter_file,
                  'r') as white_object_parameter_file:
            self.white_object_detection_parameters = json.load(
                white_object_parameter_file)
        self.bilateral_filter_parameters = self.white_object_detection_parameters.get(
            "BilateralFilter", None)
        self.clahe_parameters = self.white_object_detection_parameters.get(
            "Clahe",
            None)
        self.color_threshold_parameters = self.white_object_detection_parameters.get(
            "ColorThreshold", None)
        self.dilate_erode_parameters = self.white_object_detection_parameters.get(
            "DilateErode", None)

    def bilateral_filter(self, image):
        if self.bilateral_filter_parameters is None:
            print("Bilateral filter parameters are not defined")
            exit()
        return cv2.bilateralFilter(image,
                                   self.bilateral_filter_parameters['d'],
                                   self.bilateral_filter_parameters[
                                       'SigmaColor'],
                                   self.bilateral_filter_parameters[
                                       'SigmaSpace'],
                                   borderType=self.bilateral_filter_parameters.get(
                                       "BorderType", None))

    def adaptive_equalizer(self, image):
        if self.clahe_parameters is None:
            print("Clahe parameters are not defined")
            exit()
        clahe = cv2.createCLAHE(clipLimit=self.clahe_parameters["ClipLimit"],
                                tileGridSize=tuple(
                                    self.clahe_parameters["TileSize"]))
        if self.clahe_parameters['ColorSpace'] in ('BGR', 'RGB'):
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image[:, :, 1] = clahe.apply(image[:, :, 1])
            image[:, :, 2] = clahe.apply(image[:, :, 2])
        elif self.clahe_parameters['ColorSpace'] == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.clahe_parameters['ColorSpace'] == 'Grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def apply_dilation_erosion(self, image):
        if self.dilate_erode_parameters is None:
            print("Dilate and erode parameters are not defined")
            exit()
        kernel = np.ones(self.dilate_erode_parameters["KernelSize"], np.uint8)
        image = cv2.dilate(image,
                           kernel,
                           iterations=self.dilate_erode_parameters[
                               "DilationIteration"])
        image = cv2.erode(image,
                          kernel,
                          iterations=self.dilate_erode_parameters[
                              "ErosionIteration"])
        return image

    @staticmethod
    def _detect_threshold_limit(image_channel, histogram_param):
        arr = np.reshape(image_channel, (1, np.product(image_channel.shape)))
        hist_gray, bin_edge_gray = np.histogram(arr[0],
                                                histogram_param["Method"])
        white_bin_edge = bin_edge_gray[len(bin_edge_gray) - 2]
        white_bin_edge_max_bound = bin_edge_gray[len(bin_edge_gray) - 1]
        white_color_thresh = bin_edge_gray[len(bin_edge_gray) - 1] - histogram_param[
            "WhiteLim"]
        return white_bin_edge, white_color_thresh, white_bin_edge_max_bound

    def get_min_threshold_3channel(self, image, histogram_threshold_param):
        white_bin_edge0, thresh_lim0, white_bin_edge_max0 = self._detect_threshold_limit(image[:, :, 0],
                                                                                         histogram_threshold_param)
        white_bin_edge1, thresh_lim1, white_bin_edge_max1 = self._detect_threshold_limit(image[:, :, 1],
                                                                                         histogram_threshold_param)
        white_bin_edge2, thresh_lim2, white_bin_edge_max2 = self._detect_threshold_limit(image[:, :, 2],
                                                                                         histogram_threshold_param)
        return np.uint8([white_bin_edge0, white_bin_edge1, white_bin_edge2]),\
               np.uint8([thresh_lim0, thresh_lim1, thresh_lim2]), np.uint8([white_bin_edge_max0, white_bin_edge_max1, white_bin_edge_max2])

    def threshold_white_color(self, image):
        white_bin_edge = None
        white_max = None
        if self.color_threshold_parameters is None:
            print("Color threshold parameters are not defined")
            exit()
        if self.color_threshold_parameters["ColorSpace"] is "BGR":
            if "HistogramThreshold" in self.color_threshold_parameters:
                white_bin_edge, lower_white, white_max = self.get_min_threshold_3channel(image,
                                                              self.color_threshold_parameters[
                                                                  "HistogramThreshold"])
            elif "NormalThreshold" in self.color_threshold_parameters:
                lower_white = np.uint8(
                    self.color_threshold_parameters["NormalThreshold"][
                        "LowerLimit"])
            else:
                lower_white = np.uint8([230, 230, 230])
            upper_white = np.uint8([255, 255, 255])
            thresh = cv2.inRange(image, lower_white, upper_white)
            return thresh

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if "HistogramThreshold" in self.color_threshold_parameters:
                white_bin_edge, lower_white, white_max = self._detect_threshold_limit(image,
                                                           self.color_threshold_parameters["HistogramThreshold"])
            elif "NormalThreshold" in self.color_threshold_parameters:
                lower_white = \
                self.color_threshold_parameters["NormalThreshold"]["LowerLimit"]
            else:
                lower_white = 230
            upper_white = 255
            ret, thresh = cv2.threshold(image, lower_white, upper_white,
                                        cv2.THRESH_BINARY)
            return thresh, white_bin_edge, lower_white, white_max, image

    def white_object_detection(self, image):
        lower_white_bin = None
        considered_lower_white = None
        white_max_value = None
        #  Apply bilateral filtering in original color image
        if "BilateralFilter" in self.white_object_detection_parameters:
            image = self.bilateral_filter(image)
        # Apply adaptive clip limit equalizer or clahe for removal
        if "Clahe" in self.white_object_detection_parameters:
            image = self.adaptive_equalizer(image)
        # Threshold white object based on white color limit
        if "ColorThreshold" in self.white_object_detection_parameters:
            image, lower_white_bin, considered_lower_white, white_max_value, grayscale_image = self.threshold_white_color(image)
            # plt.subplot(111), plt.imshow(image)
            # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # cv2.waitKey(0)
        # Apply dilate and erode to eliminate noise inside
        if "DilateErode" in self.white_object_detection_parameters:
            image = self.apply_dilation_erosion(image)
            # plt.subplot(111), plt.imshow(image)
            # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # cv2.waitKey(0)
        return image, lower_white_bin, considered_lower_white, white_max_value, grayscale_image
