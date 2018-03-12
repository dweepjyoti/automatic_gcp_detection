import csv
import os

project_folder = input("Enter the image directory for detection : ")
file_count = 1
file_name = os.path.join(project_folder, 'detect_param_info' + str(file_count)+ '.csv')
detection_param_file = os.path.join(project_folder, 'detection_param_info_all.csv')
with open(file_name, newline='') as file:
    first_line = file.readline()
    print(first_line)
    with open(detection_param_file, 'w') as detection_param_file_csv:
        writer = csv.writer(detection_param_file_csv)
        writer.writerow(first_line)

while os.path.exists(file_name):
    with open(file_name, newline='') as File:
        reader = csv.reader(File)
        next(reader)
        for row in reader:
            print(row)
            with open(detection_param_file, 'a') as detection_param_file_csv:
                writer = csv.writer(detection_param_file_csv)
                writer.writerow(row)
    file_count += 1
    file_name = os.path.join(project_folder,
                             'detect_param_info' + str(file_count) + '.csv')
