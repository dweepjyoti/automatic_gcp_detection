import csv
import os
import json

project_folder = input("Enter the image directory for detection : ")
file_name = os.path.join(project_folder, 'detection_param_info_all.csv')
modified_detection_param = os.path.join(project_folder, 'modified_detection_param_info_all.csv')
CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
L_shape_detection_parameter_file = os.path.join(CONFIG_FOLDER,
                                                'Lshapedetection.json')
if not os.path.exists(L_shape_detection_parameter_file):
    print("L shape detection parameter file does not exists")
with open(L_shape_detection_parameter_file,
          'r') as L_shape_parameter_file:
    L_shape_detection_parameters = json.load(
        L_shape_parameter_file)
print(L_shape_detection_parameters)
EliminatedFrom = None

# with open(file_name, newline='') as file:
#     first_line = file.readline()
#     print(first_line)
#     with open(modified_detection_param, 'w') as detection_param_file_csv:
#         writer = csv.writer(detection_param_file_csv)
#         writer.writerow(first_line)



while os.path.exists(file_name):
    with open(file_name, newline='') as File:
        reader = csv.reader(File)
        next(reader)
        for row in reader:
            if row:
                row_items = list(row)
                row_to_write = row_items[0:52]
                # print('Row to write : {}'.format(row_to_write))
                # print('Row is : {}'.format(row_items[0]))
                corner_point_result = int(row_items[50]) - 6
                h2len_minboundrect = float(row_items[29])
                w2len_minboundrect = float(row_items[30])
                h2w_minboundrect = float(row_items[31])
                area_minboundrect = float(row_items[34])
                peri_minboundrect = float(row_items[35])
                area_actualgcp = float(row_items[36])
                peri_actualgcp = float(row_items[37])
                defect_len = float(row_items[48])
                row_to_write[4] = False
                if not 0 <= corner_point_result < L_shape_detection_parameters["CornerPoints"]["MaxDiff"]:
                    EliminatedFrom = 'corner_points'
                elif not L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Min"] < h2len_minboundrect < L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Max"]:
                    EliminatedFrom = 'height2gcplen_MinBoundRect'
                elif not L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Min"] < w2len_minboundrect < L_shape_detection_parameters["MinBoundRectTest"]["SideRatio"]["Max"]:
                    EliminatedFrom = 'width2gcplen_MinBoundRect'
                elif not L_shape_detection_parameters["MinBoundRectTest"]["HeightWidthRatio"]["Min"] < h2w_minboundrect < L_shape_detection_parameters["MinBoundRectTest"]["HeightWidthRatio"]["Max"]:
                    EliminatedFrom = 'height2width_MinBoundRect'
                elif not L_shape_detection_parameters["MinBoundRectTest"]["AreaRatio"]["Min"] < area_minboundrect < L_shape_detection_parameters["MinBoundRectTest"]["AreaRatio"]["Max"]:
                    EliminatedFrom = 'arearatio_MinBoundRect'
                elif not L_shape_detection_parameters["MinBoundRectTest"]["PerimeterRatio"]["Min"] < peri_minboundrect < L_shape_detection_parameters["MinBoundRectTest"]["PerimeterRatio"]["Max"]:
                    EliminatedFrom = 'periratio_MinBoundRect'
                elif not L_shape_detection_parameters["ActualAreaPerimeterTest"]["AreaRatio"]["Min"] < area_actualgcp < L_shape_detection_parameters["ActualAreaPerimeterTest"]["AreaRatio"]["Max"]:
                    EliminatedFrom = 'actual_area_ratio'
                elif not L_shape_detection_parameters["ActualAreaPerimeterTest"]["PerimeterRatio"]["Min"] < peri_actualgcp < L_shape_detection_parameters["ActualAreaPerimeterTest"]["PerimeterRatio"]["Max"]:
                    EliminatedFrom = 'actual_peri_ratio'
                elif not L_shape_detection_parameters["DefectRatio"]["Min"] < defect_len < L_shape_detection_parameters["DefectRatio"]["Max"]:
                    EliminatedFrom = 'max_defect2line_len_ratio'
                else:
                    row_to_write[4] = True
                    EliminatedFrom = 'Not eliminated'
                row_to_write.append(EliminatedFrom)
                # print('Eliminated from : {}'.format(EliminatedFrom))
                # print(row_items)
                with open(modified_detection_param, 'a', newline='') as detection_param_file_csv:
                    writer = csv.writer(detection_param_file_csv)
                    writer.writerow(row_to_write)
