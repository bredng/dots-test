"""
Tracks multiple objects by selecting regions of interest with OpenCV trackers
"""
import cv2
import sys
import numpy as np
from helpers import *

# Setting parameters
font = cv2.FONT_HERSHEY_SIMPLEX
all_points = []
escape_key = 27

# Prompt user to select tracker type
while True:
    try:
        print(f" Select a tracker from the following:\n \
        1. Boosting\n \
        2. MIL\n \
        3. KCF\n \
        4. TLD\n \
        5. Median flow\n \
        6. CSRT (Recommended for accuracy, low FPS throughput)\n \
        7. MOSSE (Recommended for speed, low accuracy)\n")
        selection = int(input("Selected tracker type: "))

        # Initialise tracker based on selected tracker type
        if selection == 1:
            tracker = cv2.legacy.TrackerBoosting_create()
            break
        elif selection == 2:
            tracker = cv2.legacy.TrackerMIL_create()
            break
        elif selection == 3:
            tracker = cv2.legacy.TrackerKCF_create()
            break
        elif selection == 4:
            tracker = cv2.legacy.TrackerTLD_create()
            break
        elif selection == 5:
            tracker = cv2.legacy.TrackerMedianFlow()
            break
        elif selection == 6:
            tracker = cv2.legacy.TrackerCSRT_create()
            break
        elif selection == 7:
            tracker = cv2.legacy.TrackerMOSSE_create()
            break
        else:
            print("Invalid input, please select a tracker from 1-7")
    except ValueError:
        print("Invalid input, please select a tracker from 1-7")

# Initialise multiple object tracker
trackers = cv2.legacy.MultiTracker_create()

# Retrieve video feed
cap = cv2.VideoCapture("testvid.mp4")
success, frame = cap.read()

# Prompt user to select region of interest
bbox = cv2.selectROI("Select region of interest", frame, fromCenter=False, showCrosshair=False)
# Initialise tracker with chosen region of interest and add to multi-tracker object
trackers.add(tracker, frame, bbox)

# Clear windows
cv2.destroyAllWindows

# Loop through the remaining frames
while True:
    success, frame = cap.read()
    success, boxes = trackers.update(frame)

    # End of video, break out of loop
    if frame is None:
        break

    # Update every bounding box in boxes
    for box in boxes:
        x, y, w, h = [int(dimension) for dimension in box]
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)

    # Display result
    cv2.imshow("Result", frame)
    if cv2.waitKey(30) & 0xff == escape_key:
       break

cv2.destroyAllWindows()
sys.exit()

