"""
Tracks one or more points of interest using sparse optical flow
"""
import sys
import numpy as np
import cv2
from helpers import *


# Retrieve video
cap = cv2.VideoCapture("testvid.mp4")

# Retrieve video's first frame
success, frame = cap.read()

# Convert first frame to grayscale
previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Setting parameters
lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS |
                                                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Array to hold new user-defined points to track
user_points = np.empty([0, 1, 2], dtype=np.float32)

def select_point(event, x, y, flags, params) -> None:
    """
    A callback function that saves the pixel selected by the user upon left mouse click.

    :param event: The input event
    :param x: The x coordinate of the pixel interacted with when the event occurred
    :param y: The y coordinate of the pixel interacted with when the event occurred
    :param flags: Any flags
    :param params: Any additional parameters
    """
    global point, selected_point, old_points, clicked, user_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y) # Record coordinates where left mouse click occurred
        selected_point = True
        user_points = np.empty([1, 1, 2], dtype=np.float32)
        user_points[0][0] = [x, y]
        clicked = True

# Initialise variables
selected_point = False
point = ()
old_points = np.empty([0, 1, 2], dtype=np.float32)
clicked = False

# Creating window and associating callback function with window
window_name = "Select a point to track"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_point)

# Prompt user to select point of interest
while True:
    cv2.imshow(window_name, frame)
    
    if clicked or cv2.waitKey(1) & 0xFF == escape_key:
        break

# Reset flag for future calls
clicked = False

# Clear opened windows
cv2.destroyAllWindows()

# Create blank mask
canvas = np.zeros_like(frame)
canvas = cv2.circle(canvas, point, 5, (0, 0, 255), 2)

# Loop through remaining frames
while True:
    # Get next frame
    success, frame = cap.read()

    if frame is None:
        break

    # Convert frame to grayscale
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        if len(user_points) > 0:
            old_points = np.concatenate([old_points, user_points])
            user_points = np.empty([0, 1, 2], dtype=np.float32)
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(previous, next, old_points, None,
                                                         **lk_params)

        for i, (new, old) in enumerate(zip(new_points, old_points)):
            x, y = new.ravel()
            j, k = old.ravel()

            # Draw a line between the old and new points
            canvas = cv2.line(canvas, (int(x), int(y)), (int(j), int(k)), (0, 255, 0), 3)
            # Draw a circle at the new point
            frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            frame = cv2.circle(frame, (int(j), int(k)), 5, (0, 255, 0), -1)

    key = cv2.waitKey(30) & 0xFF

    if key == s_key:
        cv2.destroyAllWindows()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_point)
        while True:
            cv2.imshow(window_name, frame)
            if clicked or cv2.waitKey(1) & 0xFF == escape_key:
                clicked = False
                canvas = cv2.circle(canvas, point, 5, (0, 0, 255), 2)
                break

    result = cv2.add(frame, canvas)
    cv2.imshow("Result", result)
    previous = next.copy()
    old_points = new_points
    if key == escape_key:
        break

cv2.destroyAllWindows()
sys.exit()
