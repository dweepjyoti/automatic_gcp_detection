import json
import os
import csv
from datetime import datetime
from glob import glob
from matplotlib import pyplot as plt

import cv2
import numpy as np
import re
import codecs

from modules.colorbaseddetector import WhiteObjectDetector
from modules.lshapedetection import LShapeDetector
from helpers.image import Image

mode = "Detection"
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
true_positive = 0
true_positive_data = {}
false_negative = 0
false_negative_data = {}
false_negative_not_in_contour_detaction = 0
false_positive = 0
false_positive_data = {}
total_images = 0
total_contours = 0
total_gcps = 0
no_images_with_gcps = 0
images_with_gcps = []

for key, value in verification_images.items():
    total_gcps += len(value)

train_result_folder = os.path.join(image_directory, 'training_result')
true_positive_dir = os.path.join(image_directory, 'TruePositive')
false_positive_dir = os.path.join(image_directory, 'FalsePositive')
false_negative_dir = os.path.join(image_directory, 'FalseNegative')

if mode == "Training":
    if not os.path.exists(train_result_folder):
        os.mkdir(train_result_folder)
else:
    if not os.path.exists(true_positive_dir):
        os.mkdir(true_positive_dir)
    if not os.path.exists(false_negative_dir):
        os.mkdir(false_negative_dir)
    if not os.path.exists(false_positive_dir):
        os.mkdir(false_positive_dir)


def verify_result(shape_detection_result, image_file, image):
    true_pos = 0
    true_positive_location = []
    false_neg = 0
    false_negative_location = []
    false_pos = 0
    false_positive_location = []
    gcp_location_list_verfication_file = verification_images[image_file].tolist()
    print(gcp_location_list_verfication_file)
    for key, value in shape_detection_result.items():
        # if key != 'TotalContours':
        gcp_location = value["GCPLocation"]
        for gcp in gcp_location_list_verfication_file:
            abs_x_diff = np.abs(gcp_location[1] - gcp[1])
            abs_y_diff = np.abs(gcp_location[0] - gcp[0])
            if abs_x_diff < 15 and abs_y_diff < 15:
                print("Detected as true positive")
                true_pos += 1
                true_positive_location.append(gcp_location)
                true_positive_file = os.path.join(true_positive_dir, os.path.split(image_file)[1])
                print(true_positive_file)
                cv2.imwrite(true_positive_file, image)
                gcp_location_list_verfication_file.remove(gcp)

    for key, value in shape_detection_result.items():
        if value["GCPLocation"] not in true_positive_location:
            print("Detected as false positive")
            false_pos += 1
            false_positive_location.append(value["GCPLocation"])
            false_positive_file = os.path.join(false_positive_dir, os.path.split(image_file)[1])
            cv2.imwrite(false_positive_file, image)

    if gcp_location_list_verfication_file:
        print("Detected as false negative")
        false_neg = len(gcp_location_list_verfication_file)
        false_negative_location = gcp_location_list_verfication_file
        false_negative_file = os.path.join(false_negative_dir, os.path.split(image_file)[1])
        cv2.imwrite(false_negative_file, image)

    return true_pos, false_pos, false_neg, true_positive_location, false_positive_location, false_negative_location


start_time = datetime.now()
for image_file in glob(os.path.join(image_directory, 'Geotagged-Images', '*.JPG')):
    file_name = str(os.path.join(proj_name, os.path.basename(image_file)))# .replace('\\', '\\\\')
    print('Reading images : {}'.format(file_name))
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    # print('cv read imgage: {}'.format(image))
    # plt.subplot(111), plt.imshow(cv2.cvtColor(image,
    #                                           cv2.COLOR_BGR2RGB))
    # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # cv2.waitKey(0)
    binary_image, lower_white, considered_white, white_max = white_detector.white_object_detection(image)
    # plt.subplot(111), plt.imshow(binary_image)
    # plt.title('Image with gcp'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # cv2.waitKey(0)
    if mode == "Training":
        if file_name not in verification_images:
            continue
        gcp_locations = verification_images[file_name]
    else:
        gcp_locations = None
    if mode == "Training":
        shape_detection_result, resulted_detection_image, total_conts = l_shape_detector.detect_l_shape_and_locate_gcp(
            binary_image, image, mode, gcp_locations)
        test_results[file_name] = shape_detection_result
        result_image_file = os.path.join(train_result_folder, file_name)
        cv2.imwrite(result_image_file, resulted_detection_image)
    elif mode == "Detection":
        shape_detection_result, resulted_detection_image, total_conts = l_shape_detector.detect_l_shape_and_locate_gcp(
            binary_image, image)
        print('Shape detection result : {}'.format(shape_detection_result))
        print(file_name)
        total_contours += total_conts
        if file_name in verification_images and shape_detection_result:
            print("Preparing for result")
            t_p, f_p, f_n, t_pos_loc, f_pos_loc, f_neg_loc = verify_result(
                shape_detection_result,
                file_name,
                resulted_detection_image)
            true_positive = true_positive + t_p
            false_positive = false_positive + f_p
            false_negative = false_negative + f_n
            if t_pos_loc:
                true_positive_data[file_name] = str(t_pos_loc)
                no_images_with_gcps += 1
                images_with_gcps.append(file_name)
            if f_pos_loc:
                false_positive_data[file_name] = str(f_pos_loc)
            if f_neg_loc:
                false_negative_data[file_name] = str(f_neg_loc)
        elif file_name not in verification_images and shape_detection_result:
            print("Detected as false positive")
            false_positive_gcp_list = []
            false_positive += len(shape_detection_result)
            false_positive_file = os.path.join(false_positive_dir,
                                               os.path.split(file_name)[1])
            cv2.imwrite(false_positive_file, resulted_detection_image)
            for key, value in shape_detection_result.items():
                # if key != 'TotalContours':
                false_positive_gcp_list.append(value["GCPLocation"])
                false_positive_data[file_name] = str(false_positive_gcp_list)
        elif file_name in verification_images and not shape_detection_result:
            print("Detected as true negative")
            false_negative_not_in_contour_detaction = false_negative_not_in_contour_detaction + len(verification_images[file_name])
            false_negative_data[file_name] = str(verification_images[
                file_name])
            false_negative_file = os.path.join(false_negative_dir,
                                              os.path.split(file_name)[1])
            cv2.imwrite(false_negative_file, resulted_detection_image)
        total_images += 1


end_time = datetime.now()
time_taken = end_time - start_time
test_results["TimeTaken"] = str(time_taken)
test_results["TotalImages"] = total_images
test_results["NoImagesWithGCPs"] = no_images_with_gcps
test_results["NoImagesWithoutGCPs"] = total_images - no_images_with_gcps
test_results["TotalContours"] = total_contours
test_results["TruePositive"] = true_positive,
test_results["TruePositiveContLoc"] = true_positive_data
test_results["FalseNegative"] = false_negative + false_negative_not_in_contour_detaction
test_results["FalseNegativeFromContour"] = false_negative
test_results["FalseNegativeNotFromContour"] = false_negative_not_in_contour_detaction
test_results["FalseNegativeContLoc"] = false_negative_data
test_results["FalsePositive"] = false_positive
test_results["FalsePositiveContLoc"] = false_positive_data
test_results["TrueNegative"] = total_contours - true_positive - false_positive - false_negative_not_in_contour_detaction
test_results["TotalGCPs"] = total_gcps
print("Test results : {}".format(test_results))

perf_check_file = os.path.join(image_directory, 'performance_gcp_detection.json')
with open(perf_check_file, 'w') as perf_check_json:
    json.dump(test_results, perf_check_json)

performance_file = os.path.join(image_directory, 'Performance.csv')
fields = ["TimeTaken", "TotalImages", "NoImagesWithGCPs",
          "NoImagesWithoutGCPs", "TotalGCPs", "TotalContours", "TruePositive",
          "FalseNegative", "FalsePositive", "TrueNegative", "FalseNegativeFromContour",
          "FalseNegativeNotFromContour", 'TruePositiveContLoc', 'FalseNegativeContLoc',
          'FalsePositiveContLoc']
# if not os.path.exists(performance_file):
with open(performance_file, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    print(test_results["TruePositive"], type(test_results["TruePositive"]))
    writer.writerow(test_results)

print("TimeTaken : {}".format(test_results["TimeTaken"]))
print("Total images : {}".format(total_images))
print("NoImagesWithGCPs : {}".format(no_images_with_gcps))
print("NoImagesWithoutGCPs: {}".format(total_images - no_images_with_gcps))
print("TotalContours : {}".format(total_contours))
print("True positive : {}".format(true_positive))
print("False negative : {}".format(false_negative+ false_negative_not_in_contour_detaction))
print("False poitive : {}".format(false_positive))
print("True negative : {}".format(total_contours - true_positive - false_positive))
test_results["Total GCPs"] = total_gcps