import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class LShapeDetector:
    """'ShapeDetector' is a class to detect 'L' shaped marker from a grayscale or binary image. Some notes:
    1) Perimeter and area of original marker is used to detect the shape.
    2) Length of one piece of L shape marker is 650mm and width is 650/4.33mm.
    3) Perimeter = 4*650 mm and area = (2/4.33 - (1/4.33)^2)*65 mm2
    Attributes:
        1) camera = camera used in photography DJI camera and Sony camera
        2) Based on camera sensor type following constants -
            a) sensor_width = (6.17, 4.55)mm for DJI and (23.5, 15.6)mm for Sony
            b) image_width  = (4000, 3000)pixels for DJI and (6000, 4000)pixels for Sony
            c) focal_length = 3.6mm for DJI and 16mm for Sony
        """

    def __init__(self, flight_data):
        CONFIG_FOLDER = os.path.join(
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            'config')
        self.L_shape_detection_parameter_file = os.path.join(CONFIG_FOLDER,
                                                             'Lshapedetection.json')
        if not os.path.exists(self.L_shape_detection_parameter_file):
            print("L shape detection parameter file does not exists")
        with open(self.L_shape_detection_parameter_file,
                  'r') as L_shape_parameter_file:
            self.L_shape_detection_parameters = json.load(
                L_shape_parameter_file)

        self.gcp_marker_property_file = os.path.join(CONFIG_FOLDER,
                                                     'gcpmarkerproperties.json')
        if not os.path.exists(self.gcp_marker_property_file):
            print("gcp marker property file does not exists")
        with open(self.gcp_marker_property_file, 'r') as gcp_marker_file:
            self.gcp_marker_properties = json.load(gcp_marker_file)

        self.height = flight_data["height"]
        self.sensor_size = flight_data["Camera"]["SensorSize"]
        self.focal_length = flight_data["Camera"]["FocalLength"]
        self.image_size = flight_data["Camera"]["ImageSize"]
        self.image_size_ratio = (self.image_size[0] * self.focal_length / (
            self.sensor_size[0] * self.height),
                                 self.image_size[1] * self.focal_length / (
                                     self.sensor_size[1] * self.height))
        print('Sensor size ratio: {}'.format(self.image_size_ratio))
        self.gcp_length, self.gcp_width, self.area, self.perimeter = self._get_marker_parameter_in_image_dimension(
            self.gcp_marker_properties)
        self.gcp_list = []

    @staticmethod
    def get_min_max_average_white_gcp_intensity(grayscale_image, cnt):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayscale_image)
        meanVal = (minVal + maxVal)/2.00
        mask = np.zeros(grayscale_image.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        # pixelpoints = np.transpose(np.nonzero(mask))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grayscale_image,
                                                           mask=mask)
        mean_val = (min_val+max_val)/2.00
        # print('Mean Value : {}'.format(mean_val))
        return min_val, max_val, mean_val, minVal, maxVal, meanVal

    def _get_marker_parameter_in_image_dimension(self, marker_parameter):
        gcp_length = marker_parameter["Length"] * min(self.image_size_ratio[0],
                                                      self.image_size_ratio[0])
        gcp_width = marker_parameter["Width"] * min(self.image_size_ratio[0],
                                                    self.image_size_ratio[0])
        gcp_area = marker_parameter["Area"] * self.image_size_ratio[0] * \
                   self.image_size_ratio[1]
        gcp_perimeter = marker_parameter["Perimeter"] * (
            self.image_size_ratio[0] + self.image_size_ratio[1])
        return gcp_length, gcp_width, gcp_area, gcp_perimeter

    def minBoundingRect_test(self, contour, contour_area, contour_perimeter):
        ((loc_x, loc_y), (w, h), angle) = cv2.minAreaRect(contour)
        if not all([h > 0.0, w > 0.0]):
            return None
        bounding_rect_area = w * h
        min_bound_rect_peri = 2 * (h + w)
        return {"CenterLocation": [loc_x, loc_y],
                "Height_MinBoundRect": h,
                "Width_MinBoundRect" : w,
                "Height2Len_MinBoundRect": h / self.gcp_length,
                "Width2Len_MinBoundRect": w / self.gcp_length,
                "HeightWidthRatio": h / w,
                "Area_MinBoundRect" : bounding_rect_area,
                "Peri_MinBoundRect" : min_bound_rect_peri,
                "AreaRatio": bounding_rect_area / contour_area,
                "PerimeterRatio": contour_perimeter / min_bound_rect_peri}

    def actual_area_and_perimeter_test(self, contour_area, contour_perimeter):
        return {"ActualAreaRatio": contour_area / self.area,
                "ActualPerimeterRatio": contour_perimeter / self.perimeter}

    @staticmethod
    def get_distance(point1, point2):
        return np.sqrt(np.sum([np.square(point1[0][0] - point2[0][0]),
                               np.square(point1[0][1] - point2[0][1])]))

    def get_gcp_location(self, contour, contour_peri, cnt_area):
        hull = cv2.convexHull(contour, returnPoints=False)
        hull_points = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull_points)
        if hull_area > 0.00:
            solidity = float(cnt_area) / hull_area
        else:
            solidity = None
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return None
        else:
            position = np.array(np.where(np.max(defects[:, :, 3])))
            start = tuple(contour[defects[position[0], 0, 0]][0])
            end = tuple(contour[defects[position[0], 0, 1]][0])
            far = tuple(contour[defects[position[0], 0, 2]][0])
            dist_start_end = self.get_distance(start, end)
            defect = defects[0][0][3] / 256.0
            ratio_defect_line_length = defect / dist_start_end
            ratio_defect_contour_peri = defect/ contour_peri
            return {"StartPoint": start,
                    "EndPoint": end,
                    "FarPoint": far,
                    "MaxDefect_ConvexHull" : defect,
                    "LineLen_ConvexHull" : dist_start_end,
                    "RatioDefectLen2LineLen": ratio_defect_line_length,
                    "RatioDefectLen2ContourPeri": ratio_defect_contour_peri,
                    "Convex_hull_area": hull_area,
                    "Solidity": solidity}


    def get_bounding_rect_parameters(self, cnt, cnt_area):
        x, y, w, h = cv2.boundingRect(cnt)
        if h>0.00:
            aspect_ratio = float(w) / h
        else:
            return None
        rect_area = w * h
        extent = float(cnt_area) / rect_area
        equiv_dia = np.sqrt(4*cnt_area/np.pi)
        return {"Height_BoundingRect" : h,
                "Width_BoundingRect" : w,
                "AspectRatio" : aspect_ratio,
                "BoundingRectArea" : rect_area,
                "Extent" : extent,
                "EqvDia" : equiv_dia}

    def detect_l_shape_and_locate_gcp(self, binary_image, original_image, gray_image, mode="Detection", gcp_location=None):
        binary_image_dup = original_image
        # Contour detection on binary image
        image, contours, hierarchy = cv2.findContours(binary_image,
                                                      cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        # Initialize shape based detection result
        L_shape_detection_result_main = {}
        l_count = 1
        # for each contour which will match the L shape
        for i in range(len(contours)):
            L_shape_detection_result = {}
            contour = contours[i]

            # get area and perimeter of contour
            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, 0.025 * contour_perimeter,
                                       True)
            # point noise removal
            if not (contour_area > 0.0 and contour_perimeter > 0.0):
                continue

            # Check concavity of contour. L shape always will be concave
            if cv2.isContourConvex(contour):
                continue

            # minimum bounding rect test
            bounding_rect_test_result = self.minBoundingRect_test(contour,
                                                                  contour_area,
                                                                  contour_perimeter)
            actual_area_perimeter_test_result = self.actual_area_and_perimeter_test(
                contour_area, contour_perimeter)
            corner_point_result = len(contour)-6
            rect_result = self.get_bounding_rect_parameters(contour,
                                                            contour_area)
            gcp_loc_result = self.get_gcp_location(contour, contour_perimeter, contour_area)
            if gcp_loc_result is None:
                continue
            min_white_val, max_white_val, mean_white_val, min_intensity_image, max_intensity_image, mean_intensity_image = self.get_min_max_average_white_gcp_intensity(
                gray_image, contour)
            # print(min_white_val, max_white_val, mean_white_val)
            # plt.subplot(111), plt.imshow(gray_image)
            # plt.title('gray_image'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # cv2.waitKey(0)
            L_shape_detection_result["GCP_length"] = self.gcp_length
            L_shape_detection_result["GCP_width"] = self.gcp_width
            L_shape_detection_result["GCP_area"] = self.area
            L_shape_detection_result["GCP_perimeter"] = self.perimeter
            # Contour properties
            L_shape_detection_result["ContourArea"] = contour_area
            L_shape_detection_result["ContourPeri"] = contour_perimeter
            # white property result
            L_shape_detection_result["WhiteIntensity_Min"] = min_white_val
            L_shape_detection_result["WhiteIntensity_Max"] = max_white_val
            L_shape_detection_result["WhiteIntensity_Mean"] = mean_white_val
            # image intensities
            L_shape_detection_result["ImageIntensity_Min"] = min_intensity_image
            L_shape_detection_result["ImageIntensity_Max"] = max_intensity_image
            L_shape_detection_result["ImageIntensity_Mean"] = mean_intensity_image
            # Corner point result
            L_shape_detection_result["CornerPoints"] = corner_point_result + 6
            # min bound rect results
            L_shape_detection_result["Center_MinBoundRect"] = \
                bounding_rect_test_result["CenterLocation"]
            L_shape_detection_result["Height_MinBoundRect"] = \
                bounding_rect_test_result["Height_MinBoundRect"]
            L_shape_detection_result["Width_MinBoundRect"] = \
                bounding_rect_test_result["Width_MinBoundRect"]
            L_shape_detection_result["Height2Len_MinBoundRect"] = \
                bounding_rect_test_result["Height2Len_MinBoundRect"]
            L_shape_detection_result["Width2Len_MinBoundRect"] = \
                bounding_rect_test_result["Width2Len_MinBoundRect"]
            L_shape_detection_result["Height2Width_MinBoundrRect"] = \
                bounding_rect_test_result["HeightWidthRatio"]
            L_shape_detection_result["Area_MinBoundRect"] = \
                bounding_rect_test_result["Area_MinBoundRect"]
            L_shape_detection_result["Peri_MinBoundRect"] = \
                bounding_rect_test_result["Peri_MinBoundRect"]
            L_shape_detection_result["AreaRatio_MinBoundRect"] = \
                bounding_rect_test_result["AreaRatio"]
            L_shape_detection_result["PeriRatio_MinBoundRect"] = \
                bounding_rect_test_result["PerimeterRatio"]
            # actual gcp area and perimeter parameters
            L_shape_detection_result["AreaRatio_ActualGCP"] = \
                actual_area_perimeter_test_result["ActualAreaRatio"]
            L_shape_detection_result["PeriRatio_ActualGCP"] = \
                actual_area_perimeter_test_result["ActualPerimeterRatio"]
            # bounding rect result
            L_shape_detection_result["Height_BoundingRect"] = rect_result[
                "Height_BoundingRect"]
            L_shape_detection_result["Width_BoundingRect"] = rect_result[
                "Width_BoundingRect"]
            L_shape_detection_result["AspectRatio_BoundingRect"] = \
                rect_result["AspectRatio"]
            L_shape_detection_result["Area_BoundingRect"] = rect_result[
                "BoundingRectArea"]
            L_shape_detection_result["Extent_BoundingRect"] = rect_result[
                "Extent"]
            L_shape_detection_result["EqvDia_BoundingRect"] = rect_result[
                "EqvDia"]
            # gcp detection result
            L_shape_detection_result["MaxDefect_ConvexHull"] = \
                gcp_loc_result["MaxDefect_ConvexHull"]
            L_shape_detection_result["LineLen_ConvexHull"] = gcp_loc_result[
                "LineLen_ConvexHull"]
            L_shape_detection_result["RatioDefectLen2LineLen"] = \
                gcp_loc_result["RatioDefectLen2LineLen"]
            L_shape_detection_result["RatioDefectLen2ContourPeri"] = \
                gcp_loc_result["RatioDefectLen2ContourPeri"]
            L_shape_detection_result["Convex_hull_area"] = gcp_loc_result[
                "Convex_hull_area"]
            L_shape_detection_result["Solidity"] = gcp_loc_result[
                "Solidity"]
            L_shape_detection_result["GCPLocation"] = \
                gcp_loc_result["FarPoint"][0].tolist()
            L_shape_detection_result["Contour"] = contour
            cv2.putText(original_image,
                        str(l_count),
                        (
                            int(bounding_rect_test_result["CenterLocation"][0]),
                            int(bounding_rect_test_result["CenterLocation"][
                                    1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA)
            cv2.drawContours(original_image, [contour], 0, (0, 255, 0), 3)
            L_shape_detection_result["DetectionResult"] = False
            # if mode == 'Training':
            #     inside_corner = gcp_loc_result["FarPoint"][0].tolist()
            #     # print("Inside corner : {}".format(inside_corner))
            #     # for gcp_location in gcp_locations:
            #     # print("gcp location: {}".format(gcp_location))
            #     abs_x_diff = np.abs(gcp_location[1] - inside_corner[1])
            #     abs_y_diff = np.abs(gcp_location[0] - inside_corner[0])
            #     if abs_x_diff < 20 and abs_y_diff < 20:
            #         L_shape_detection_result = {}
            #         # gcp parameters
            #         L_shape_detection_result["GCP_length"] = self.gcp_length
            #         L_shape_detection_result["GCP_width"] = self.gcp_width
            #         L_shape_detection_result["GCP_area"] = self.area
            #         L_shape_detection_result["GCP_perimeter"] = self.perimeter
            #         # Contour properties
            #         L_shape_detection_result["ContourArea"] = contour_area
            #         L_shape_detection_result["ContourPeri"] = contour_perimeter
            #         # Corner point result
            #         L_shape_detection_result["CornerPoints"] = corner_point_result + 6
            #         # min bound rect results
            #         L_shape_detection_result["Center_MinBoundRect"]= bounding_rect_test_result["CenterLocation"]
            #         L_shape_detection_result["Height_MinBoundRect"] = bounding_rect_test_result["Height_MinBoundRect"]
            #         L_shape_detection_result["Width_MinBoundRect"] = bounding_rect_test_result["Width_MinBoundRect"]
            #         L_shape_detection_result["Height2Len_MinBoundRect"] = bounding_rect_test_result["Height2Len_MinBoundRect"]
            #         L_shape_detection_result["Width2Len_MinBoundRect"] = bounding_rect_test_result["Width2Len_MinBoundRect"]
            #         L_shape_detection_result["Height2Width_MinBoundrRect"] = bounding_rect_test_result["HeightWidthRatio"]
            #         L_shape_detection_result["Area_MinBoundRect"] = bounding_rect_test_result["Area_MinBoundRect"]
            #         L_shape_detection_result["Peri_MinBoundRect"] = bounding_rect_test_result["Peri_MinBoundRect"]
            #         L_shape_detection_result["AreaRatio_MinBoundRect"] = bounding_rect_test_result["AreaRatio"]
            #         L_shape_detection_result["PeriRatio_MinBoundRect"] = bounding_rect_test_result["PerimeterRatio"]
            #         # actual gcp area and perimeter parameters
            #         L_shape_detection_result["AreaRatio_ActualGCP"] = actual_area_perimeter_test_result["ActualAreaRatio"]
            #         L_shape_detection_result["PeriRatio_ActualGCP"] = actual_area_perimeter_test_result["ActualPerimeterRatio"]
            #         # bounding rect result
            #         L_shape_detection_result["Height_BoundingRect"] = rect_result["Height_BoundingRect"]
            #         L_shape_detection_result["Width_BoundingRect"] = rect_result["Width_BoundingRect"]
            #         L_shape_detection_result["AspectRatio_BoundingRect"] = rect_result["AspectRatio"]
            #         L_shape_detection_result["Area_BoundingRect"] = rect_result["BoundingRectArea"]
            #         L_shape_detection_result["Extent_BoundingRect"] = rect_result["Extent"]
            #         L_shape_detection_result["EqvDia_BoundingRect"] = rect_result["EqvDia_BoundingRect"]
            #         # gcp detection result
            #         L_shape_detection_result["MaxDefect_ConvexHull"] = gcp_loc_result["MaxDefect_ConvexHull"]
            #         L_shape_detection_result["LineLen_ConvexHull"] = gcp_loc_result["LineLen_ConvexHull"]
            #         L_shape_detection_result["RatioDefectLen2LineLen"] = gcp_loc_result["RatioDefectLen2LineLen"]
            #         L_shape_detection_result["RatioDefectLen2ContourPeri"] = gcp_loc_result["RatioDefectLen2ContourPeri"]
            #         L_shape_detection_result["Convex_hull_area"] = gcp_loc_result["Convex_hull_area"]
            #         L_shape_detection_result["Solidity"] = gcp_loc_result["Solidity"]
            #         L_shape_detection_result["GCPLocation"] = gcp_loc_result["FarPoint"][0].tolist()
            #         L_shape_detection_result["Contour"] = contour
            #         # cv2.putText(original_image,
            #         #             str(l_count),
            #         #             (
            #         #                 int(bounding_rect_test_result[
            #         #                         "CenterLocation"][0]),
            #         #                 int(bounding_rect_test_result[
            #         #                         "CenterLocation"][
            #         #                         1])),
            #         #             cv2.FONT_HERSHEY_SIMPLEX,
            #         #             4,
            #         #             (0, 0, 255),
            #         #             3,
            #         #             cv2.LINE_AA)
            #         cv2.drawContours(original_image, [contour], 0,
            #                          (0, 255, 0), 3)
            #         # print("LshapeResult:{}".format(L_shape_detection_result))
            #         return L_shape_detection_result, binary_image_dup
            #
            # else:
            if not 0 <= corner_point_result < self.L_shape_detection_parameters["CornerPoints"]["MaxDiff"]:
                L_shape_detection_result["EliminatedFrom"] = 'corner_points'
            elif not self.L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Min"] < bounding_rect_test_result["Height2Len_MinBoundRect"] < self.L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'height2gcplen_MinBoundRect'
            elif not self.L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Min"] < bounding_rect_test_result["Width2Len_MinBoundRect"] < self.L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'width2gcplen_MinBoundRect'
            elif not self.L_shape_detection_parameters["MinBoundRectTest"]["HeightWidthRatio"]["Min"] < bounding_rect_test_result["HeightWidthRatio"] < self.L_shape_detection_parameters["MinBoundRectTest"]["HeightWidthRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'height2width_MinBoundRect'
            elif not self.L_shape_detection_parameters["MinBoundRectTest"]["AreaRatio"]["Min"] < bounding_rect_test_result["AreaRatio"] < self.L_shape_detection_parameters["MinBoundRectTest"]["AreaRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'arearatio_MinBoundRect'
            elif not self.L_shape_detection_parameters["MinBoundRectTest"]["PerimeterRatio"]["Min"] < bounding_rect_test_result["PerimeterRatio"] < self.L_shape_detection_parameters["MinBoundRectTest"]["PerimeterRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'periratio_MinBoundRect'
            elif not self.L_shape_detection_parameters["ActualAreaPerimeterTest"]["AreaRatio"]["Min"] < actual_area_perimeter_test_result["ActualAreaRatio"] < self.L_shape_detection_parameters["ActualAreaPerimeterTest"]["AreaRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'actual_area_ratio'
            elif not self.L_shape_detection_parameters["ActualAreaPerimeterTest"]["PerimeterRatio"]["Min"] < actual_area_perimeter_test_result["ActualPerimeterRatio"] < self.L_shape_detection_parameters["ActualAreaPerimeterTest"]["PerimeterRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'actual_peri_ratio'
            elif not self.L_shape_detection_parameters["DefectRatio"]["Min"] < gcp_loc_result["RatioDefectLen2LineLen"] < self.L_shape_detection_parameters["DefectRatio"]["Max"]:
                L_shape_detection_result["EliminatedFrom"] = 'max_defect2line_len_ratio'
            else:
                L_shape_detection_result["DetectionResult"] = True
                L_shape_detection_result["EliminatedFrom"] = 'Not eliminated'
            # if l_count == 43:
            #     print(bounding_rect_test_result["AreaRatio"])
            #     print(type(bounding_rect_test_result["AreaRatio"]))
            #     print(type())
            #     print(L_shape_detection_result["Center_MinBoundRect"])
            #     print(L_shape_detection_result["DetectionResult"])
            #     print(L_shape_detection_result["EliminatedFrom"])


            L_shape_detection_result_main[
                "Cont" + str(l_count)] = L_shape_detection_result
            l_count += 1
        # L_shape_detection_result_main['TotalContours'] = len(contours)

        return L_shape_detection_result_main, original_image, len(contours)