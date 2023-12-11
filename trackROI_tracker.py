"""
Tracks an object by selecting a region of interest with OpenCV trackers
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

# Retrieve video feed
cap = cv2.VideoCapture("testvid.mp4")
success, frame = cap.read()

# Prompt user to select region of interest
bbox = cv2.selectROI("Select region of interest",frame, fromCenter=False, showCrosshair=False)
# Initialise tracker with chosen region of interest
tracker.init(frame, bbox)

# Update all points array with first point
all_points.append(get_centre(bbox))

# Clear windows
cv2.destroyAllWindows()

# Calculate time between each frame
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1/video_fps

# Loop through the remaining frames
while True:
    timer = cv2.getTickCount()
    success, frame = cap.read()
    success, bbox = tracker.update(frame)

    # End of video, break out of loop
    if frame is None:
        break

    # If object is successfully being tracked, update box as normal
    if success:
        draw_box(frame, bbox, font)
        new_point = get_centre(bbox)
        magnitude_text = (new_point[0] + 30, new_point[1] - 30)
        angle_text = (new_point[0] + 30, new_point[1] - 60)
        velocity_text = (new_point[0] + 30, new_point[1] - 90)
        
        # Draw vector
        v_magnitude, v_angle = calculate_vector(all_points[-1], new_point)
        velocity = v_magnitude/frame_time
        cv2.arrowedLine(frame, all_points[-1], new_point, (255, 0, 0), 2)

        cv2.putText(frame, "Magnitude:" + str(v_magnitude), magnitude_text, font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Angle:" + str(v_angle), angle_text, font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Velocity:" + str(velocity), velocity_text, font, 0.7, (255, 0, 0), 2)
        all_points.append(new_point)
    else:
        # Otherwise update text to indicate object has been lost 
        cv2.putText(frame, "Lost", (100, 75), font, 0.7, (0, 0, 255), 2)

    # Draw rectangle to contain status and FPS
    cv2.rectangle(frame, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(frame, "FPS:", (20, 40), font, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "Status:", (20, 75), font, 0.7, (255, 0, 255), 2)

    # Calculate FPS and display
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, str(int(fps)), (75, 40), font, 0.7, (20, 20, 230), 2)

    # Display result
    cv2.imshow("Result", frame)
    if cv2.waitKey(30) & 0xff == escape_key:
       break

cv2.destroyAllWindows()
sys.exit()
