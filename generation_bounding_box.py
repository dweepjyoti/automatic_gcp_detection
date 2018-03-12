import csv
import os
import re
import json

import cv2
import numpy as np

from matplotlib import pyplot as plt

from helpers.image import Image
from modules.colorbaseddetector import WhiteObjectDetector
from modules.lshapedetection import LShapeDetector

project_directory = input("Enter the project directory : ") # E:\Images\GCP_MARKERS\L&T-NIMZ_Flight_M1_F1.2
proj_name = os.path.split(project_directory)[1]
print('Project Name : {}'.format(proj_name))
geo_tagged_image_dir = os.path.join(project_directory,
                                    "Geotagged-Images")
if not os.path.exists(geo_tagged_image_dir):
    print("Geotagged-Images folder is not there in project directory")
gcp_location_csv_file = os.path.join(project_directory, "GCP_location.csv") # "GCP_location.csv"
gcp_file = {}
try:
    with open(gcp_location_csv_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for project_n_image_name, gcp_locations in csv_reader:
            file_name = os.path.basename(project_n_image_name)
            gcp_points = np.array([np.int32(float(point)) for point in
                                   re.findall("\d+\.\d+", gcp_locations)])
            gcp_locations = gcp_points.reshape(int(len(gcp_points) / 2), 2)
            gcp_file[file_name] = gcp_locations
except FileNotFoundError:
    print("GCP_location.csv is not there in project directory")

print('File name dict : {}'.format(gcp_file))
flight_data_file = os.path.join(project_directory, 'flightparameters.json')
if not os.path.exists(flight_data_file):
    print("Flight data file does not exists")
with open(flight_data_file, 'r') as flight_param_file:
    flight_data = json.load(flight_param_file)
white_detector = WhiteObjectDetector()
l_shape_detector = LShapeDetector(flight_data)
bounding_box_info = os.path.join(project_directory,
                                    'bound_box_info.csv')
bounding_box_field = ["FileName", "x1", "y1", "x2", "y2", "IsGCP"]
with open(bounding_box_info, 'w') as bound_box_file:
    writer = csv.DictWriter(bound_box_file, fieldnames=bounding_box_field,
                            lineterminator='\n')
    writer.writeheader()
def write_to_csvs(bound_box_dict):
    with open(bounding_box_info, 'a') as bound_box_file:
        writer = csv.DictWriter(bound_box_file,
                                fieldnames=bounding_box_field,
                                lineterminator='\n')
        writer.writerow(bound_box_dict)
for file in gcp_file.items():
    print(file[0])
    abs_file_path = os.path.join(project_directory,'Geotagged-Images', file[0])
    image = cv2.imread(abs_file_path, cv2.IMREAD_COLOR)
    if image is None:
        continue
    binary_image, lower_white, considered_white, white_max, gray_image = white_detector.white_object_detection(
        image)
    image1, contours, hierarchy = cv2.findContours(binary_image,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        L_shape_detection_result = {}
        contour = contours[i]
        # Check concavity of contour. L shape always will be concave
        if cv2.isContourConvex(contour):
            continue
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            continue
        position = np.array(np.where(np.max(defects[:, :, 3])))
        start = tuple(contour[defects[position[0], 0, 0]][0])
        end = tuple(contour[defects[position[0], 0, 1]][0])
        far = tuple(contour[defects[position[0], 0, 2]][0])
        max_def_point = far[0].tolist()
        for location in file[1]:
            print('GCP location from file : {}'.format([location[1], location[0]]))
            abs_x_diff = np.abs(max_def_point[1] - location[1])
            abs_y_diff = np.abs(max_def_point[0] - location[0])
            if abs_x_diff < 15 and abs_y_diff < 15:
                bound_box = {}
                print('Far : {}'.format(far[0].tolist()))
                x, y, w, h = cv2.boundingRect(contour)
                # bounding_box_location = [x, y, x+w, y+h]
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # plt.show()
                # cv2.waitKey(0)
                file_name = os.path.relpath(abs_file_path, project_directory)
                image_name = os.path.join(proj_name, file_name)
                print(image_name)
                bound_box["FileName"] = image_name
                bound_box["x1"] = x
                bound_box["y1"] = y
                bound_box["x2"] = x+w
                bound_box["y2"] = y+h
                bound_box['IsGCP'] = True
                write_to_csvs(bound_box)

