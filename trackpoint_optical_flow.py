"""
Tracks an object by selecting a point of interest with sparse optical flow
"""
import sys
import numpy as np
import cv2


# Retrieve video
cap = cv2.VideoCapture("testvid1.mp4")

# Retrieve video's first frame
succ, frame = cap.read()

# Convert first frame to grayscale
previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Setting parameters
lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS |
                                                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Function that allows user to set a point of interest
def select_point(event, x, y, flags, params):
    global point, selected_point, old_points, clicked
    # Record coordinates of mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)
        clicked = True

# Initialise variables
selected_point = False
point = ()
old_points = ([[]])
clicked = False

window_name = "Select a point to track"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_point)

# Prompt user to select point of interest
while True:
    cv2.imshow(window_name, frame)
    
    if clicked or cv2.waitKey(1) & 0xFF == 13:
        break

cv2.destroyAllWindows()

# Create blank mask
canvas = np.zeros_like(frame)

# Loop through remaining frames
while True:
    succ, frame = cap.read()
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(previous, next, old_points, None,
                                                         **lk_params)

        previous = next.copy()
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        # Draw a line between the old and new points
        canvas = cv2.line(canvas, (int(x), int(y)), (int(j), int(k)), (0, 255, 0), 3)
        # Draw a circle at the new point
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    result = cv2.add(frame, canvas)
    cv2.imshow("Result", result)
    if cv2.waitKey(30) & 0xFF == 13:
        break

cv2.destroyAllWindows()
sys.exit()
