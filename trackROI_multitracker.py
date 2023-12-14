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
            tracker = "boosting"
            break
        elif selection == 2:
            tracker = "mil"
            break
        elif selection == 3:
            tracker = "kcf"
            break
        elif selection == 4:
            tracker = "tld"
            break
        elif selection == 5:
            tracker = "medianflow"
            break
        elif selection == 6:
            tracker = "csrt"
            break
        elif selection == 7:
            tracker = "mosse"
            break
        else:
            print("Invalid input, please select a tracker from 1-7")
    except ValueError:
        print("Invalid input, please select a tracker from 1-7")

# Initialise multiple object tracker
trackers = cv2.legacy.MultiTracker_create()

# Retrieve video feed
cap = cv2.VideoCapture("testmulti.mp4")
success, frame = cap.read()

# Prompt user to select region of interest
cv2.namedWindow("Select regions of interest", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select regions of interest", 960, 540)
boxes = cv2.selectROIs("Select regions of interest", frame, fromCenter=False, showCrosshair=False)
# Initialise tracker with chosen regions of interest and add to multi-tracker object
for box in boxes:
    trackers.add(create_tracker(tracker), frame, box)
    all_points.append([get_centre(box)])

# Clear windows
cv2.destroyAllWindows()

# Initialise mask
draw_mask = np.zeros_like(frame)

# Loop through the remaining frames
while True:
    success, frame = cap.read()
    success, boxes = trackers.update(frame)

    # End of video, break out of loop
    if frame is None:
        break
    
    # Update every bounding box in boxes
    for i in range(len(boxes)):
        draw_box(frame, boxes[i])
        new_point = get_centre(boxes[i])

        # Draw vector
        v_magnitude, v_angle = calculate_vector(all_points[i][-1], new_point)
        cv2.arrowedLine(frame, all_points[i][-1], new_point, (255, 0, 0), 2)
        draw_mask = cv2.line(draw_mask, all_points[i][-1], new_point, (255, 0, 0), 2)

        cv2.putText(frame, str(i), new_point, font, 0.7, (0, 255, 0), 2)
        all_points[i].append(new_point)
    
    img = cv2.add(frame, draw_mask)
    
    # Read user input
    key = cv2.waitKey(30) & 0xff

    # Add a new tracker 
    if key == s_key:
        cv2.destroyAllWindows()
        # Prompt user to select region of interest
        boxes = cv2.selectROIs("Select regions of interest", frame, fromCenter=False, showCrosshair=False)
        # Initialise tracker with chosen region of interest and add to multi-tracker object
        for box in boxes:
            trackers.add(create_tracker(tracker), frame, box)
            all_points.append([get_centre(box)])
        cv2.destroyAllWindows()

    # Display result
    cv2.imshow("Result", img)
    if key == escape_key:
       break

cv2.destroyAllWindows()
sys.exit()

