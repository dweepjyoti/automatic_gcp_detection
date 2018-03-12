import json
import os
import csv
from datetime import datetime
from glob import glob
from matplotlib import pyplot as plt
import srtm

import cv2
import numpy as np
import re
import codecs

from modules.colorbaseddetector import WhiteObjectDetector
from modules.lshapedetection import LShapeDetector
from helpers.image import Image

mode = "Detection"
srtm_data = srtm.get_data()
image_directory = input("Enter the image directory for detection : ")
proj_name = os.path.split(image_directory)[1]

flight_data_file = os.path.join(image_directory, 'flightparameters.json')
if not os.path.exists(flight_data_file):
    print("Flight data file does not exists")
with open(flight_data_file, 'r') as flight_param_file:
    flight_data = json.load(flight_param_file)

verification_file = os.path.join(image_directory, 'GCP_location.csv')
if not os.path.exists(verification_file):
    print("Verification file does not exists")

verification_images = {}
with open(verification_file, newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for project_n_image_name, gcp_locations in csv_reader:
        gcp_points = np.array([np.int32(float(point)) for point in
                               re.findall("\d+\.\d+", gcp_locations)])
        gcp_locations = gcp_points.reshape(int(len(gcp_points) / 2), 2)
        verification_images[str(project_n_image_name).replace('\\\\','\\')] = gcp_locations



print('Verification images: {}'.format(verification_images))
white_detector = WhiteObjectDetector()
l_shape_detector = LShapeDetector(flight_data)

test_results = {}
total_images = 0
total_contours = 0
total_gcps = 0
no_images_with_gcps = 0
images_with_gcps = []
for key, value in verification_images.items():
    total_gcps += len(value)

fields_contour_info = ["Number", "ImageName", "CntNo", "CntCenter", "IsGCP", "DetectedGCP",
                       "CntIntensity_Min", "CntIntensity_Max", "CntIntensity_Mean", "CntArea",
                       "CntPeri", "GCP_Length", "GCP_Area", "GCP_Peri", "MinBoundRect_Height",
                       "MinBoundRect_Width", "BoundingRect_Height",
                       "BoundingRect_Width", "EqvDia", "ConvexHull_Area",
                       "ConvexHull_MaxDefect", "ConvexHull_LineLen",
                       "CornerPoints"]
fields_for_detection_param = ["Number", "ImageName", "CntNo", "CntCenter", "IsGCP", "DetectedGCP",
                              "Brightness", "HSV_Brightness", "Blur", "CntIntensity_Min",
                              "CntIntensity_Max", "CntIntensity_Mean", "ImageIntensity_Min",
                              "ImageIntensity_Max", "ImageIntensity_Mean", "WhiteIntensityUsed_Min",
                              "WhiteIntensityUsed_Max", "CntArea", "CntPeri", "GCP_Length",
                              "GCP_Width", "GCP_Area", "GCP_Peri", "RelAlt", "CameraModel",
                              "FocalLength", "SensorHeight", "SensorWidth", "MinBoundRect_Height",
                              "MinBoundRect_Width", "MinBoundRect_Height2Len", "MinBoundRect_Width2Len",
                              "MinBoundrRect_Height2Width", "MinBoundRect_Area", "MinBoundRect_Peri",
                              "MinBoundRect_AreaRatio", "MinBoundRect_PeriRatio", "ActualGCP_AreaRatio",
                              "ActualGCP_PeriRatio", "BoundingRect_Height", "BoundingRect_Width",
                              "BoundingRect_AspectRatio", "BoundingRect_Area", "BoundingRect_Extent",
                              "EqvDia", "ConvexHull_Area", "ConvexHull_MaxDefect", "ConvexHull_LineLen",
                              "ConvexHull_Solidity", "RatioDefectLen2LineLen", "RatioDefectLen2ContourPeri",
                              "CornerPoints", "GCP_Location", "EliminatedFrom"]


line_count_csv = 1
binary_image_folder = os.path.join(image_directory, 'binary_image')
if not os.path.exists(binary_image_folder):
    os.mkdir(binary_image_folder)

def write_to_csvs(result):
    global line_count_csv
    for key, value in result.items():
        cnt_info = {}
        detect_param = {}
        cnt_info_csv = os.path.join(image_directory,
                                    'cont_info.csv')
        detect_param_csv = os.path.join(image_directory,
                                    'detect_param_info.csv')
        if not os.path.exists(cnt_info_csv):
            with open(cnt_info_csv, 'w') as cnt_info_file:
                writer = csv.DictWriter(cnt_info_file, fieldnames=fields_contour_info,
                                        lineterminator='\n')
                writer.writeheader()
        if not os.path.exists(detect_param_csv):
            with open(detect_param_csv, 'w') as detect_param_file:
                writer = csv.DictWriter(detect_param_file, fieldnames=fields_for_detection_param,
                                        lineterminator='\n')
                writer.writeheader()
        print('Result : {}'.format(result[key]))
        cnt_info["Number"] = line_count_csv
        cnt_info["ImageName"] = result[key].get("FileName", None)
        cnt_info["CntNo"] = key
        cnt_info["CntCenter"] = result[key].get("Center_MinBoundRect", None)
        cnt_info["IsGCP"] = result[key].get("GroundTruth", None)
        cnt_info["DetectedGCP"] = result[key].get("DetectionResult", None)
        cnt_info["CntIntensity_Min"] = result[key].get("WhiteIntensity_Min", None)
        cnt_info["CntIntensity_Max"] = result[key].get("WhiteIntensity_Max", None)
        cnt_info["CntIntensity_Mean"] = result[key].get("WhiteIntensity_Mean", None)
        cnt_info["CntArea"] = result[key].get("ContourArea", None)
        cnt_info["CntPeri"] = result[key].get("ContourPeri", None)
        cnt_info["GCP_Length"] = result[key].get("GCP_length", None)
        cnt_info["GCP_Area"] = result[key].get("GCP_area", None)
        cnt_info["GCP_Peri"] = result[key].get("GCP_perimeter", None)
        cnt_info["MinBoundRect_Height"] = result[key].get("Height_MinBoundRect", None)
        cnt_info["MinBoundRect_Width"] = result[key].get("Width_MinBoundRect", None)
        cnt_info["BoundingRect_Height"] = result[key].get("Height_BoundingRect", None)
        cnt_info["BoundingRect_Width"] = result[key].get("Width_BoundingRect", None)
        cnt_info["EqvDia"] = result[key].get("EqvDia_BoundingRect", None)
        cnt_info["ConvexHull_Area"] = result[key].get("Convex_hull_area", None)
        cnt_info["ConvexHull_MaxDefect"] = result[key].get("MaxDefect_ConvexHull", None)
        cnt_info["ConvexHull_LineLen"] = result[key].get("LineLen_ConvexHull", None)
        cnt_info["CornerPoints"] = result[key].get("CornerPoints", None)

        detect_param["Number"] = line_count_csv
        detect_param["ImageName"] = result[key].get("FileName", None)
        detect_param["CntNo"] = key
        detect_param["CntCenter"] = result[key].get("Center_MinBoundRect", None)
        detect_param["IsGCP"] = result[key].get("GroundTruth", None)
        detect_param["DetectedGCP"] = result[key].get("DetectionResult", None)
        detect_param["Brightness"] = result[key].get("Brightness", None)
        detect_param["HSV_Brightness"] = result[key].get("HSV_Brightness", None)
        detect_param["Blur"] = result[key].get("Blur", None)
        detect_param["CntIntensity_Min"] = result[key].get("WhiteIntensity_Min", None)
        detect_param["CntIntensity_Max"] = result[key].get("WhiteIntensity_Max", None)
        detect_param["CntIntensity_Mean"] = result[key].get("WhiteIntensity_Mean", None)
        detect_param["ImageIntensity_Min"] = result[key].get("ImageIntensity_Min", None)
        detect_param["ImageIntensity_Max"] = result[key].get("ImageIntensity_Max", None)
        detect_param["ImageIntensity_Mean"] = result[key].get("ImageIntensity_Mean", None)
        detect_param["WhiteIntensityUsed_Min"] = result[key].get("WhiteIntensityUsed_Min", None)
        detect_param["WhiteIntensityUsed_Max"] = result[key].get("WhiteIntensityUsed_Max", None)
        detect_param["CntArea"] = result[key].get("ContourArea", None)
        detect_param["CntPeri"] = result[key].get("ContourPeri", None)
        detect_param["GCP_Length"] = result[key].get("GCP_length", None)
        detect_param["GCP_Width"] = result[key].get("GCP_width", None)
        detect_param["GCP_Area"] = result[key].get("GCP_area", None)
        detect_param["GCP_Peri"] = result[key].get("GCP_perimeter", None)
        detect_param["RelAlt"] = result[key].get("RelAlt", None)
        detect_param["CameraModel"] = result[key].get("CameraModel", None)
        detect_param["FocalLength"] = result[key].get("FocalLength", None)
        detect_param["SensorHeight"] = result[key].get("SensorHeight", None)
        detect_param["SensorWidth"] = result[key].get("SensorWidth", None)
        detect_param["MinBoundRect_Height"] = result[key].get("Height_MinBoundRect", None)
        detect_param["MinBoundRect_Width"] = result[key].get("Width_MinBoundRect", None)
        detect_param["MinBoundRect_Height2Len"] = result[key].get("Height2Len_MinBoundRect", None)
        detect_param["MinBoundRect_Width2Len"] = result[key].get("Width2Len_MinBoundRect", None)
        detect_param["MinBoundrRect_Height2Width"] = result[key].get("Height2Width_MinBoundrRect", None)
        detect_param["MinBoundRect_Area"] = result[key].get("Area_MinBoundRect", None)
        detect_param["MinBoundRect_Peri"] = result[key].get("Peri_MinBoundRect", None)
        detect_param["MinBoundRect_AreaRatio"] = result[key].get("AreaRatio_MinBoundRect", None)
        detect_param["MinBoundRect_PeriRatio"] = result[key].get("PeriRatio_MinBoundRect", None)
        detect_param["ActualGCP_AreaRatio"] = result[key].get("AreaRatio_ActualGCP", None)
        detect_param["ActualGCP_PeriRatio"] = result[key].get("PeriRatio_ActualGCP", None)
        detect_param["BoundingRect_Height"] = result[key].get("Height_BoundingRect", None)
        detect_param["BoundingRect_Width"] = result[key].get("Width_BoundingRect", None)
        detect_param["BoundingRect_AspectRatio"] = result[key].get("AspectRatio_BoundingRect", None)
        detect_param["BoundingRect_Area"] = result[key].get("Area_BoundingRect", None)
        detect_param["BoundingRect_Extent"] = result[key].get("Extent_BoundingRect", None)
        detect_param["EqvDia"] = result[key].get("EqvDia_BoundingRect", None)
        detect_param["ConvexHull_Area"] = result[key].get("Convex_hull_area", None)
        detect_param["ConvexHull_MaxDefect"] = result[key].get("MaxDefect_ConvexHull", None)
        detect_param["ConvexHull_LineLen"] = result[key].get("LineLen_ConvexHull", None)
        detect_param["ConvexHull_Solidity"] = result[key].get("Solidity", None)
        detect_param["RatioDefectLen2LineLen"] = result[key].get("RatioDefectLen2LineLen", None)
        detect_param["RatioDefectLen2ContourPeri"] = result[key].get("RatioDefectLen2ContourPeri", None)
        detect_param["CornerPoints"] = result[key].get("CornerPoints", None)
        detect_param["GCP_Location"] = result[key].get("GCPLocation", None)
        detect_param["EliminatedFrom"] = result[key].get("EliminatedFrom", None)
        line_count_csv += 1
        with open(cnt_info_csv, 'a') as cnt_info_file:
            writer = csv.DictWriter(cnt_info_file,
                                    fieldnames=fields_contour_info,
                                    lineterminator='\n')
            writer.writerow(cnt_info)
        with open(detect_param_csv, 'a') as detect_param_file:
            writer = csv.DictWriter(detect_param_file,
                                    fieldnames=fields_for_detection_param,
                                    lineterminator='\n')
            writer.writerow(detect_param)



def verify_result(shape_detection_result, image_file, image, considered_white, white_max, im):
    true_positive_location = []
    failed_cont_detect = 0
    gcp_location_list_verfication_file = verification_images[image_file].tolist()
    lat = im.lat
    lng = im.lng
    alt = im.abs_alt
    relative_altitude = im.relative_alt
    if relative_altitude is None:
        relative_altitude = alt - srtm_data.get_elevation(
            latitude=lat,
            longitude=lng)
    # print(gcp_location_list_verfication_file)
    for key, value in shape_detection_result.items():
        gcp_location = value["GCPLocation"]
        for gcp in gcp_location_list_verfication_file:
            abs_x_diff = np.abs(gcp_location[1] - gcp[1])
            abs_y_diff = np.abs(gcp_location[0] - gcp[0])
            if abs_x_diff < 15 and abs_y_diff < 15:
                true_positive_location.append(gcp_location)
                shape_detection_result[key]["GroundTruth"] = True
                shape_detection_result[key]["FileName"] = file_name
                shape_detection_result[key]["RelAlt"] = relative_altitude
                shape_detection_result[key]["CameraModel"] = im.camera_model
                shape_detection_result[key]["FocalLength"] = im.focal_length
                shape_detection_result[key]["SensorHeight"] = im.sensor_height
                shape_detection_result[key]["SensorWidth"] = im.sensor_width
                shape_detection_result[key]["Brightness"] = im.brightness
                shape_detection_result[key]["HSV_Brightness"] = im.hsv_brightness
                shape_detection_result[key]["Blur"] = im.blur
                shape_detection_result[key]["WhiteIntensityUsed_Min"] = considered_white
                shape_detection_result[key]["WhiteIntensityUsed_Max"] = white_max
                print("Detected as true positive")
                gcp_location_list_verfication_file.remove(gcp)
    for key, value in shape_detection_result.items():
        if value["GCPLocation"] not in true_positive_location:
            shape_detection_result[key]["GroundTruth"] = False
            shape_detection_result[key]["FileName"] = file_name
            shape_detection_result[key]["RelAlt"] = relative_altitude
            shape_detection_result[key]["CameraModel"] = im.camera_model
            shape_detection_result[key]["FocalLength"] = im.focal_length
            shape_detection_result[key]["SensorHeight"] = im.sensor_height
            shape_detection_result[key]["SensorWidth"] = im.sensor_width
            shape_detection_result[key]["Brightness"] = im.brightness
            shape_detection_result[key]["HSV_Brightness"] = im.hsv_brightness
            shape_detection_result[key]["Blur"] = im.blur
            shape_detection_result[key]["WhiteIntensityUsed_Min"] = considered_white
            shape_detection_result[key]["WhiteIntensityUsed_Max"] = white_max
    if gcp_location_list_verfication_file:
        failed_cont_detect += 1
        Cont_Detection_failed = {}
        print("Detected as false negative")
        Cont_Detection_failed["GroundTruth"] = True
        Cont_Detection_failed["DetectedGCP"] = False
        Cont_Detection_failed["FileName"] = file_name
        Cont_Detection_failed["RelAlt"] = relative_altitude
        Cont_Detection_failed["CameraModel"] = im.camera_model
        Cont_Detection_failed["FocalLength"] = im.focal_length
        Cont_Detection_failed["SensorHeight"] = im.sensor_height
        Cont_Detection_failed["SensorWidth"] = im.sensor_width
        Cont_Detection_failed["Brightness"] = im.brightness
        Cont_Detection_failed["HSV_Brightness"] = im.hsv_brightness
        Cont_Detection_failed["Blur"] = im.blur
        Cont_Detection_failed["WhiteIntensityUsed_Min"] = considered_white
        Cont_Detection_failed["WhiteIntensityUsed_Max"] = white_max
        shape_detection_result['FailedContDetection'+str(failed_cont_detect)] = Cont_Detection_failed
    return shape_detection_result


start_time = datetime.now()
for image_file in glob(os.path.join(image_directory, 'Geotagged-Images', '*.JPG')):
    cont_param_gen_fail = 0
    image_basename = os.path.basename(image_file)
    file_name = str(os.path.join(proj_name, os.path.basename(image_file)))# .replace('\\', '\\\\')
    print('Reading images : {}'.format(file_name))
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    im = Image(image_file)
    lat = im.lat
    lng = im.lng
    alt = im.abs_alt
    relative_altitude = im.relative_alt
    if relative_altitude is None:
        relative_altitude = alt - srtm_data.get_elevation(
            latitude=lat,
            longitude=lng)
    binary_image, lower_white, considered_white, white_max, gray_image = white_detector.white_object_detection(image)
    binary_image_file = os.path.join(binary_image_folder, image_basename)
    cv2.imwrite(binary_image_file, binary_image)
    shape_detection_result, resulted_detection_image, total_conts = l_shape_detector.detect_l_shape_and_locate_gcp(
        binary_image, image, gray_image)
    # print('Shape detection result : {}'.format(shape_detection_result))
    total_contours += total_conts
    if file_name in verification_images and shape_detection_result:
        print("Preparing for result")
        shape_detection_result = verify_result(
            shape_detection_result,
            file_name,
            resulted_detection_image,
            considered_white,
            white_max,
            im)
    elif file_name not in verification_images and shape_detection_result:
        for key, value in shape_detection_result.items():
            shape_detection_result[key]["GroundTruth"] = False
            shape_detection_result[key]["FileName"] = file_name
            shape_detection_result[key]["RelAlt"] = relative_altitude
            shape_detection_result[key]["CameraModel"] = im.camera_model
            shape_detection_result[key]["FocalLength"] = im.focal_length
            shape_detection_result[key]["SensorHeight"] = im.sensor_height
            shape_detection_result[key]["SensorWidth"] = im.sensor_width
            shape_detection_result[key]["Brightness"] = im.brightness
            shape_detection_result[key]["HSV_Brightness"] = im.hsv_brightness
            shape_detection_result[key]["Blur"] = im.blur
            shape_detection_result[key]["WhiteIntensityUsed_Min"] = considered_white
            shape_detection_result[key]["WhiteIntensityUsed_Max"] = white_max
    elif file_name in verification_images and not shape_detection_result:
        gcp_location_list_verfication_file = verification_images[
            file_name].tolist()
        for gcp in gcp_location_list_verfication_file:
            cont_param_gen_fail += 1
            Cont_gen_failed = {}
            print("Detected as false negative")
            Cont_gen_failed["GroundTruth"] = True
            Cont_gen_failed["DetectedGCP"] = False
            Cont_gen_failed["FileName"] = file_name
            Cont_gen_failed["RelAlt"] = relative_altitude
            Cont_gen_failed["CameraModel"] = im.camera_model
            Cont_gen_failed["FocalLength"] = im.focal_length
            Cont_gen_failed["SensorHeight"] = im.sensor_height
            Cont_gen_failed["SensorWidth"] = im.sensor_width
            Cont_gen_failed["Brightness"] = im.brightness
            Cont_gen_failed["HSV_Brightness"] = im.hsv_brightness
            Cont_gen_failed["Blur"] = im.blur
            Cont_gen_failed["WhiteIntensityUsed_Min"] = considered_white
            Cont_gen_failed["WhiteIntensityUsed_Max"] = white_max
            shape_detection_result['FailedContDetection'+str(cont_param_gen_fail)] = Cont_gen_failed
    write_to_csvs(shape_detection_result)
    total_images += 1


end_time = datetime.now()
time_taken = end_time - start_time
test_results["TimeTaken"] = str(time_taken)
test_results["TotalImages"] = total_images
test_results["NoImagesWithGCPs"] = no_images_with_gcps
test_results["NoImagesWithoutGCPs"] = total_images - no_images_with_gcps
test_results["TotalContours"] = total_contours
test_results["TotalGCPs"] = total_gcps
print("Test results : {}".format(test_results))

# perf_check_file = os.path.join(image_directory, 'performance_gcp_detection.json')
# with open(perf_check_file, 'w') as perf_check_json:
#     json.dump(test_results, perf_check_json)
#
# performance_file = os.path.join(image_directory, 'Performance.csv')
# fields = ["TimeTaken", "TotalImages", "NoImagesWithGCPs",
#           "NoImagesWithoutGCPs", "TotalGCPs", "TotalContours", "TruePositive",
#           "FalseNegative", "FalsePositive", "TrueNegative", "FalseNegativeFromContour",
#           "FalseNegativeNotFromContour", 'TruePositiveContLoc', 'FalseNegativeContLoc',
#           'FalsePositiveContLoc']
# # if not os.path.exists(performance_file):
# with open(performance_file, 'w') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator='\n')
#     writer.writeheader()
#     print(test_results["TruePositive"], type(test_results["TruePositive"]))
#     writer.writerow(test_results)

print("TimeTaken : {}".format(test_results["TimeTaken"]))
print("Total images : {}".format(total_images))
print("NoImagesWithGCPs : {}".format(no_images_with_gcps))
print("NoImagesWithoutGCPs: {}".format(total_images - no_images_with_gcps))
print("TotalContours : {}".format(total_contours))
print("ContoursInDetection : {}".format(line_count_csv))
test_results["Total GCPs"] = total_gcps
