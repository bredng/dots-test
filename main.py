import numpy as np
import cv2 
import time
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
cap = cv2.VideoCapture("testvid.mp4")

succ, first_frame = cap.read()

# Prompt user to select region of interest
x, y, w, h = cv2.selectROI("Select region of interest", first_frame, True, False)

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

print("")
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

    if cv2.waitKey(30) & 0xFF == 13:
        break

cv2.destroyAllWindows()

# succ, thres = cv2.threshold(gray_ROI, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(mask, 1, 2)

ROI_cnt = contours[0]

M = cv2.moments(mask)

# Get the centre coordinates of the object of interest
cX = int(M["m10"] / M["m00"]) + x
cY = int(M["m01"] / M["m00"]) + y

# # Convert the first frame to grayscale
# previous = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(previous, mask = None, **feature_params)

# while True:
#     frame_start = time.time()
#     succ, second_frame = cap.read()
#     next = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

#     pts0, st, err = cv2.calcOpticalFlowPyrLK(previous, next, )

#     if not succ:
#         print("Couldn't retrieve frame, ending...")
#         break
    
#     frame_end = time.time()




