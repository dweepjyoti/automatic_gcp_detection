import csv
import os
from glob import glob

import cv2
from matplotlib import pyplot as plt

project_directory = input("Enter the image directory of project : ")
image_directory = os.path.join(project_directory, "Geotagged-Images")
gcp_location_file = os.path.join(project_directory, 'GCP_location.csv')
gcp_locations = []
gcp_image_dict = {}
list_draw_points = []
previous_file = None
fields = ["FileName", "GCPLocation"]
if not os.path.exists(gcp_location_file):
    with open(gcp_location_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()


def write_to_csv(image_name, gcp_location_point):
    if gcp_location_point:
        with open(gcp_location_file, 'a') as csv_file_data:
            writer = csv.DictWriter(csv_file_data,
                                    fieldnames=fields,
                                    lineterminator='\n')
            writer.writerow({fields[0]: image_name,
                             fields[1]: gcp_location_point
                             })


def handle_mouse_press(event):
    if event.dblclick:
        plt.plot(event.xdata, event.ydata, "o", markerfacecolor='b',
                 markeredgecolor='b', markersize=5.00)
        fig.canvas.draw()
        gcp_locations.append([event.xdata, event.ydata])
    if event.button == 3 and gcp_locations:
        plt.plot(gcp_locations[-1][0], gcp_locations[-1][1], "o",
                 markerfacecolor='r', markeredgecolor='r', markersize=5.00)
        fig.canvas.draw()
        gcp_locations.pop()


def press(event):
    global gcp_locations
    global list_draw_points
    if event.key == 'n':
        plt.close(fig)
    if event.key == 't':
        mng.full_screen_toggle()


for file in glob(os.path.join(image_directory, '*JPG')):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    fig = plt.figure()
    plt.title('Geotag Image: ' + str(file)),
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90)
    ax = fig.add_subplot(111), plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
    plt.yticks([]), plt.xticks([]),
    fig.canvas.mpl_connect('button_press_event', handle_mouse_press)
    fig.canvas.mpl_connect('key_press_event', press)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    folder_name = os.path.split(os.path.split(image_directory)[0])[1]
    file_name = os.path.basename(file)
    file_name_to_store = os.path.join(folder_name, file_name)
    # gcp_image_dict[file_name_to_store] = {"GCPLocation": gcp_locations}
    write_to_csv(file_name_to_store, gcp_locations)
    gcp_locations = []
    list_draw_points = []
