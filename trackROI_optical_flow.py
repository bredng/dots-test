"""
Tracks an object by selecting a region of interest with sparse optical flow
"""

import numpy as np
import sys
import cv2 
from helpers import *


# Setting parameters
lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

feature_params = dict(maxCorners = 20, 
                      qualityLevel = 0.3, 
                      minDistance = 7, blockSize = 7)

# Array to store all the calculated velocities
velocities = []

# Frame capture
cap = cv2.VideoCapture("testvid1.mp4")

succ, first_frame = cap.read()

# Prompt user to select region of interest
x, y, w, h = cv2.selectROI("Select region of interest", first_frame, fromCenter=False, showCrosshair=False)

ROI_img = get_subimage(first_frame, x, y, w, h)

cv2.imshow("e", ROI_img)

# Find centroid of object in region of interest

# Prompt user to select HSV range
# Create window
window_name = "Select HSV range"
cv2.namedWindow(window_name)

# Create trackbar sliders
cv2.createTrackbar("HMin", window_name, 0, 179, lambda x:x)
cv2.createTrackbar("SMin", window_name, 0, 255, lambda x:x)
cv2.createTrackbar("VMin", window_name, 0, 255, lambda x:x)
cv2.createTrackbar("HMax", window_name, 0, 179, lambda x:x)
cv2.createTrackbar("SMax", window_name, 0, 255, lambda x:x)
cv2.createTrackbar("VMax", window_name, 0, 255, lambda x:x)

# Set default values of maximum trackbars
cv2.setTrackbarPos("HMax", window_name, 179)
cv2.setTrackbarPos("SMax", window_name, 255)
cv2.setTrackbarPos("VMax", window_name, 255)

# Initialise counters to keep track of values
hMin = sMin = vMin = hMax = sMax = vMax = 0

output = ROI_img

print(f"\nAdjust the sliders until only the object of interest is visible, then press ENTER button!")
while True:
    # Retrieve current trackbar positions
    hMin = cv2.getTrackbarPos("HMin", window_name)
    sMin = cv2.getTrackbarPos("SMin", window_name)
    vMin = cv2.getTrackbarPos("VMin", window_name)

    hMax = cv2.getTrackbarPos("HMax", window_name)
    sMax = cv2.getTrackbarPos("SMax", window_name)
    vMax = cv2.getTrackbarPos("VMax", window_name)

    # Set minimum and maximum HSV values
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Apply HSV range to image and threshold
    HSV_ROI = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV_ROI, lower, upper)
    output = cv2.bitwise_and(HSV_ROI, HSV_ROI, mask=mask)

    # Display output image
    cv2.imshow(window_name, output)

    # Wait for 'enter' to escape
    if cv2.waitKey(30) & 0xFF == 13:
        break

# Clear windows
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(mask, 1, 2)

ROI_cnt = contours[0]

M = cv2.moments(ROI_cnt)

# Get the centre coordinates of the object of interest
cX = int(M["m10"] / M["m00"]) + x
cY = int(M["m01"] / M["m00"]) + y

# Convert the first frame to grayscale
previous = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = np.float32([(cX, cY)]).reshape(-1, 1, 2)

# Create a mask image for drawing purposes 
draw_mask = np.zeros_like(first_frame) 

while True:
    succ, second_frame = cap.read()

    if not succ:
        print("Couldn't retrieve frame, ending...")
        break

    next = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous, next, p0, None, **lk_params)

    # Select points
    good_new = p1
    good_old = p0

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new,  
                                       good_old)): 
        a, b = new.ravel()
        c, d = old.ravel()

        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)

        draw_mask = cv2.line(draw_mask, (a, b), (c, d), 
                        (0, 255, 0), 2) 
          
        second_frame = cv2.circle(second_frame, (a, b), 5, 
                           (0, 255, 0), -1) 
          
    img = cv2.add(second_frame, draw_mask) 
  
    cv2.imshow("Result", img) 
      
    if cv2.waitKey(30) & 0xff == ord('q'):
       break
  
    previous = next.copy() 
    p0 = good_new.reshape(-1, 1, 2) 

cv2.destroyAllWindows()
sys.exit()
