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

# Find centroid of object in region of interest
gray_ROI = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)

succ, thres = cv2.threshold(gray_ROI, 127, 255, cv2.THRESH_BINARY)

M = cv2.moments(thres)

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




