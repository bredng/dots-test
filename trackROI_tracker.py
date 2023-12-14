"""
Tracks an object by selecting a region of interest with OpenCV trackers
"""
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


# Setting parameters
font = cv2.FONT_HERSHEY_SIMPLEX

# Arrays to hold output data
all_points = []
x_displacement = [0]
z_displacement = [0]
time = [0]

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
ROI_win_name = "Select region of interest"
cv2.namedWindow(ROI_win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(ROI_win_name, 960, 540)
bbox = cv2.selectROI(ROI_win_name, frame, fromCenter=False, showCrosshair=False)
# Initialise tracker with chosen region of interest
tracker.init(frame, bbox)

# Update all points array with first point
init_centre = get_centre(bbox)
all_points.append(init_centre)
init_x = init_centre[0]
init_z = init_centre[1]

# Clear windows
cv2.destroyAllWindows()

# Calculate time between each frame
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1/video_fps

# Loop through the remaining frames
while True:
    success, frame = cap.read()
    success, bbox = tracker.update(frame)

    # End of video, break out of loop
    if frame is None:
        break

    # If object is successfully being tracked, update box as normal
    if success:
        draw_box(frame, bbox)
        new_point = get_centre(bbox)
        
        # Draw vector
        v_magnitude, v_angle = calculate_vector(all_points[-1], new_point)
        velocity = v_magnitude/frame_time
        cv2.arrowedLine(frame, all_points[-1], new_point, (255, 0, 0), 2)

        # Show vector information
        cv2.putText(frame, str(round(v_magnitude, 3)), (140, 40), font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, str(round(v_angle, 3)), (100, 75), font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Tracking", (100, 110), font, 0.7, (0, 255, 0), 2)

        # Update data
        all_points.append(new_point)
        Dx = new_point[0] - init_x
        Dz = init_z - new_point[1]
        new_time = time[-1] + frame_time
        x_displacement.append(Dx)
        z_displacement.append(Dz)
        time.append(new_time)
    else:
        # Otherwise update text to indicate object has been lost 
        cv2.putText(frame, "Lost", (100, 110), font, 0.7, (0, 0, 255), 2)

    # Draw rectangle to contain status and vector information
    cv2.rectangle(frame, (15, 15), (230, 125), (255, 0, 255), 2)
    cv2.putText(frame, "Magnitude:", (20, 40), font, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "Angle:", (20, 75), font, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "Status:", (20, 110), font, 0.7, (255, 0, 255), 2)

    # Display result
    cv2.imshow("Result", frame)
    if cv2.waitKey(30) & 0xff == escape_key:
       break

cv2.destroyAllWindows()

# Creating a figure to visualise data
fig = plt.figure()

# First subplot: Time vs displacement in the x-direction
plt.subplot(2, 2, 1)
plt.plot(time, x_displacement)
plt.title("Time vs displacement in x-direction")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (pixels)")

# Second subplot: Time vs displacement in the z-direction
plt.subplot(2, 2, 2)
plt.plot(time, z_displacement)
plt.title("Time vs displacement in z-direction")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (pixels)")

# Third subplot: Path of the tracked object
plt.subplot(2, 2, (3, 4))
plt.plot(x_displacement, z_displacement)
plt.title("Path of the tracked object")
plt.xlabel("Displacement in the x-direction (pixels)")
plt.ylabel("Displacement in the y-direction (pixels)")

# Display subplots
plt.show()

# Exit the program
sys.exit()