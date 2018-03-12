import csv
import os
import re

import cv2
import numpy as np
import srtm

from matplotlib import pyplot as plt

from helpers.image import Image
from modules.colorbaseddetector import WhiteObjectDetector
from modules.lshapedetection import LShapeDetector

project_directory = input("Enter the project directory : ") # E:\Images\GCP_MARKERS\L&T-NIMZ_Flight_M1_F1.2
geo_tagged_image_dir = os.path.join(project_directory,
                                    "Geotagged-Images")
if not os.path.exists(geo_tagged_image_dir):
    print("Geotagged-Images folder is not there in project directory")
gcp_location_csv_file = os.path.join(project_directory, "GCP_location.csv") # "GCP_location.csv"
if not os.path.exists(gcp_location_csv_file):
    print("GCP_location.csv is not there in project directory")
srtm_data = srtm.get_data()
data_sample_collection_csv = os.path.join(project_directory,
                                          'TrainingSample.csv')
fields = ["FileName", "GCPLocation", "Lat", "Lng", "Alt", "RelAlt", "CameraModel",
          "FocalLength", "SensorHeight", "SensorWidth", "Brightness",
          "HSV_Brightness", "Blur", "ImageHeight", "ImageWidth", "GCP_Length",
          "GCP_Width", "GCP_Area", "GCP_Peri", "WhiteColorMinValue",
          "UsedWhiteColorMinValue", "WhiteMaxIntFromHist", "WhiteMinIntImage",
          "ContourArea", "ContourPeri", "CornerPoints", "MinBoundRect_Height",
          "MinBoundRect_Width", "MinBoundRect_RatioHeight",
          "MinBoundRect_RatioWidth", "MinBoundRect_RatioHeight2Width",
          "MinBoundRect_AreaRatioContour2Rect", "MinBoundRect_PeriRatioContour2Rect",
          "ActualGCP_AreaRatio", "ActualGCP_PeriRatio", "HullMaxDefect_DefectLen2LineLen",
          "HullMaxDefect_DefectLen2ContPeri", "GCPLOcation_ByDetection", "Contour"]
if not os.path.exists(data_sample_collection_csv):
    with open(data_sample_collection_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()

def create_roi(opencv_image, location, image_size):
    print(location)
    min_y = location[0] - 30 if (location[0] - 30) > 0 else 0
    max_y = location[0] + 30 if (location[0] + 30) < image_size[0] else image_size[0]
    min_x = location[1] - 30 if (location[1] - 30) > 0 else 0
    max_x = location[1] + 30 if (location[1] + 30) < image_size[1] else image_size[1]
    roi_image = opencv_image[min_x:max_x, min_y:max_y]
    # print(roi_image)
    grayscale_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    arr = np.reshape(grayscale_roi, (1, np.product(grayscale_roi.shape)))
    hist, bin_edges = np.histogram(arr[0], 'doane', normed=True)
    #
    # print(hist)
    # print(bin_edges)
    # print(bin_edges[len(bin_edges)-2])
    plt.subplot(211), plt.imshow(grayscale_roi)
    plt.title('ROI'), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.hist(arr[0], 'doane', normed=True)
    plt.title('Histogram plot'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)
    return roi_image


def get_cv_operation_data(opencv_image, required_image_information,
                          gcp_coordinates_in_image):
    intensities_in_contour = []
    # roi_opencv_image = create_roi(opencv_image, gcp_coordinates_in_image, required_image_information["Camera"]["ImageSize"])
    white_detector = WhiteObjectDetector()
    l_shape_detector = LShapeDetector(required_image_information)
    binary_image, lower_white, white_threshold_lower_value, white_max_value = white_detector.white_object_detection(opencv_image)
    # print(gcp_coordinates_in_image)
    shape_detection_result, resulted_detection_image = l_shape_detector.detect_l_shape_and_locate_gcp(
        binary_image,
        opencv_image,
        mode="Training",
        gcp_location=gcp_coordinates_in_image)
    grayscale_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    # print('Contour: {}'.format(shape_detection_result["Contour"]))
    intensities_in_contour = [grayscale_image[v[0][1], v[0][0]] for v in shape_detection_result["Contour"]]
    # print(intensities_in_contour)
    min_white_intensity = min(intensities_in_contour)
    # plt.subplot(111), plt.imshow(cv2.cvtColor(resulted_detection_image,
    #                                           cv2.COLOR_BGR2RGB))
    # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # cv2.waitKey(0)
    print(shape_detection_result)
    return shape_detection_result, lower_white, white_threshold_lower_value, white_max_value, min_white_intensity


def write_to_csv(gcp_detecttion_result):
    with open(data_sample_collection_csv, 'a') as csv_file_data:
        writer = csv.DictWriter(csv_file_data,
                                fieldnames=fields,
                                lineterminator='\n')
        writer.writerow(gcp_detecttion_result)


# for root, dirs, files in os.walk(combined_project_directory):
#     if "GCP_location.csv" in files:
#         project_directory = root
#         geo_tagged_image_dir = os.path.join(root, geo_tagged_image_dir_name)
#         print(geo_tagged_image_dir)
#         image_gcp_location_csv = os.path.join(project_directory,
#                                               'GCP_location.csv')
result_for_each_gcp = {}
with open(gcp_location_csv_file, newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for project_n_image_name, gcp_locations in csv_reader:
        gcp_points = np.array([np.int32(float(point)) for point in
                               re.findall("\d+\.\d+", gcp_locations)])
        gcp_locations = gcp_points.reshape(int(len(gcp_points) / 2), 2)
        project_name = os.path.split(project_n_image_name)[0]
        image_file = os.path.split(project_n_image_name)[1]
        absolute_path_image_file = os.path.join(geo_tagged_image_dir,
                                                image_file)
        if not os.path.exists(absolute_path_image_file):
            continue
        print(project_name, image_file, type(gcp_locations))
        image = Image(absolute_path_image_file)
        print("Collecting data for image : {}".format(absolute_path_image_file))
        for gcp_location in gcp_locations:
            result_for_each_gcp["FileName"] = project_n_image_name
            result_for_each_gcp["GCPLocation"] = gcp_location
            result_for_each_gcp["Lat"] = image.lat
            result_for_each_gcp["Lng"] = image.lng
            result_for_each_gcp["Alt"] = image.abs_alt
            xmp_data = image.read_xmp()
            relative_altitude = image.relative_alt
            if relative_altitude is None:
                relative_altitude = result_for_each_gcp[
                                        "Alt"] - srtm_data.get_elevation(
                    latitude=result_for_each_gcp["Lat"],
                    longitude=result_for_each_gcp["Lng"])
            result_for_each_gcp["RelAlt"] = relative_altitude
            result_for_each_gcp["CameraModel"] = image.camera_model
            result_for_each_gcp["FocalLength"] = image.focal_length
            result_for_each_gcp["SensorHeight"] = image.sensor_height
            result_for_each_gcp["SensorWidth"] = image.sensor_width
            result_for_each_gcp["Brightness"] = image.brightness
            result_for_each_gcp["HSV_Brightness"] = image.hsv_brightness
            result_for_each_gcp["Blur"] = image.blur
            cv_image = cv2.imread(absolute_path_image_file,
                                  cv2.IMREAD_COLOR)

            # plt.subplot(111), plt.imshow(cv2.cvtColor(cv_image,
            #                                           cv2.COLOR_BGR2RGB))
            # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # cv2.waitKey(0)
            result_for_each_gcp["ImageHeight"] = cv_image.shape[0]
            result_for_each_gcp["ImageWidth"] = cv_image.shape[1]
            camera_data = {"SensorSize": [result_for_each_gcp["SensorHeight"],
                                          result_for_each_gcp["SensorWidth"]],
                           "FocalLength": result_for_each_gcp["FocalLength"],
                           "ImageSize": [result_for_each_gcp["ImageHeight"],
                                         result_for_each_gcp["ImageWidth"]]}
            flight_data = {"height": relative_altitude*1000.0,
                           "Camera": camera_data}
            detection_result, lower_white_value, considered_threshold_value, white_max_value, min_white_intensity = get_cv_operation_data(cv_image,
                                                     flight_data,
                                                     gcp_location)
            print('Detection Result: {}'.format(detection_result))
            result_for_each_gcp["GCP_Length"] = detection_result["GCPLength"]
            result_for_each_gcp["GCP_Width"] = detection_result["GCPWidth"]
            result_for_each_gcp["GCP_Area"] = detection_result["GCPArea"]
            result_for_each_gcp["GCP_Peri"] = detection_result["GCPPeri"]
            result_for_each_gcp["WhiteColorMinValue"] = lower_white_value
            result_for_each_gcp["UsedWhiteColorMinValue"] = considered_threshold_value
            result_for_each_gcp["WhiteMaxIntFromHist"] = white_max_value
            result_for_each_gcp["WhiteMinIntImage"] = min_white_intensity
            result_for_each_gcp["ContourArea"] = detection_result["ContourArea"]
            result_for_each_gcp["ContourPeri"] = detection_result["ContourPeri"]
            result_for_each_gcp["CornerPoints"] = detection_result["CornerPointsDetection"]
            result_for_each_gcp["MinBoundRect_Height"] = detection_result["HeightWidthMinBoundRect"][0]
            result_for_each_gcp["MinBoundRect_Width"] = detection_result["HeightWidthMinBoundRect"][1]
            result_for_each_gcp["MinBoundRect_RatioHeight"] = detection_result["SideLengthsMinBoundRect"][0]
            result_for_each_gcp["MinBoundRect_RatioWidth"] = detection_result["SideLengthsMinBoundRect"][1]
            result_for_each_gcp["MinBoundRect_RatioHeight2Width"] = detection_result["HeightWidthRatioMinBoundRect"]
            result_for_each_gcp["MinBoundRect_AreaRatioContour2Rect"] = detection_result["AreaRatioContourMinBoundRect"]
            result_for_each_gcp["MinBoundRect_PeriRatioContour2Rect"] = detection_result["PerimeterRatioContourMinBoundRect"]
            result_for_each_gcp["ActualGCP_AreaRatio"] = detection_result["ActualGCPAreaRatio"]
            result_for_each_gcp["ActualGCP_PeriRatio"] = detection_result[
                "ActualGCPPerimeterRatio"]
            result_for_each_gcp["HullMaxDefect_DefectLen2LineLen"] = detection_result[
                "RatioDefectLen2LineLen"]
            result_for_each_gcp["HullMaxDefect_DefectLen2ContPeri"] = detection_result["RatioDefectLen2ContourPeri"]
            result_for_each_gcp["GCPLOcation_ByDetection"] = detection_result["GCPLocation"]
            result_for_each_gcp["Contour"] = detection_result["Contour"]
            print("Result for gcp :{}".format(result_for_each_gcp))
            write_to_csv(result_for_each_gcp)
