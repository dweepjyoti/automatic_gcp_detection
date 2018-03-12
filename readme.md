Project Title:
---------------
Detect a white L shaped GCP automatically in a photo and return the location of 
the GCP corner location in pixels. Save the GCP images and corresponding 
location in pixels in a CSV file.

Getting Started:
----------------
This is developed in Windows10 machine. It should still run for other OS.

**Prerequisites:**
To run this script, you should have following things running in your PC.

- Python3.6.x

**Installing and running**
To install this project try following commands
- git clone [Repository link in gitlab]
- pip install -r path/to/requirements.txt

To run this code:
- Run main.py
- It will ask for the folder location of drone images where marker needs to be find.

**Contents of drone image folder**
- Images from drone camera
- gcpimages.json: This file contains the information of number of total gcps
and images having gcps and gcp location manually rovided
It looks like:
{
	"TotalGCPs" : 43,
	"DJI_0041.JPG": [[2420, 2032]],
	"DJI_0042.JPG": [[2428, 2435]],
	"DJI_0043.JPG": [[2521, 2998]]
}
- flightparameters.json: This file contains parameters of camera like image size,
sensor size, focal length and approximate altitude or height of drone.
It looks like:
{
	"Camera":{
		"SensorSize":[4.62, 6.16],
		"ImageSize":[3000, 4000],
		"FocalLength":3.6
	},	
	"height":62100
}

**Return contents of this code**
- Folder having result images in categories TruePositive, TrueNegative and
FalsePositive with same name in drone image directory having images of results.
- A json file named perfcheckfile.json in the same directory: It will contain a
python dict having following keys and format
"TotalImage": 5,
  "Total GCPs": 3,
  "TimeTaken": "0:04:54.826607",
  "TruePositive": {
    "Count": 2,
    "Results": {
      "DJI_0041.JPG": [
        [
          2420,
          2033
        ]
      ],
      "DJI_0042.JPG": [
        [
          2428,
          2437
        ]
      ]
    }
  },
  "TrueNegative": {
    "Count": 2,
    "Results": {
      "DJI_0083.JPG": [
        [
          2333,
          572
        ]
      ],
      "DJI_0111": []
    }
  },
  "FalsePositive": {
    "Count": 0,
    "Results": {}
  }