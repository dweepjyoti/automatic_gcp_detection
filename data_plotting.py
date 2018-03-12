import csv
import os
from matplotlib import pyplot as plt

project_folder = input("Enter the image directory for detection : ")
file_name = os.path.join(project_folder, 'detect_param_info_all.csv')

if not os.path.exists(file_name):
    print("File does not exists")
true_pos_intensity_avg = []
true_pos_white_max = []
true_pos_white_avg = []

false_pos_intensity_avg = []
false_pos_white_max = []
false_pos_white_avg = []

true_neg_intensity_avg = []
true_neg_white_max = []
true_neg_white_avg = []

with open(file_name, newline='') as File:
    reader = csv.reader(File)
    next(reader)
    for row in reader:
        row_items = list(row)
        if row_items[4] == 'TRUE' and row_items[3] == 'TRUE':
            print('Detected item : {}'.format(row_items[4]))
            true_pos_intensity_avg.append(row_items[13])
            true_pos_white_max.append(row_items[9])
            true_pos_white_avg.append(row_items[10])
        if row_items[4] == 'TRUE' and row_items[3] == 'FALSE':
            print('Detected item : {}'.format(row_items[4]))
            false_pos_intensity_avg.append(row_items[13])
            false_pos_white_max.append(row_items[9])
            false_pos_white_avg.append(row_items[10])
        if row_items[4] == 'FALSE' and row_items[3] == 'TRUE':
            print('Detected item : {}'.format(row_items[4]))
            true_neg_intensity_avg.append(row_items[13])
            true_neg_white_max.append(row_items[9])
            true_neg_white_avg.append(row_items[10])

plt.figure(1)
plt.scatter(true_pos_intensity_avg, true_pos_white_max, c='b')
plt.scatter(false_pos_intensity_avg, false_pos_white_max, c='r')
plt.scatter(true_neg_intensity_avg, true_neg_white_max, c='g')
plt.show()
