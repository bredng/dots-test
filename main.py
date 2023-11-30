import numpy as np
import cv2

cap = cv2.VideoCapture("testvid.mp4")

ret, frame1 = cap.read()

# Setting parameters
winSize = cv2.Size(15, 15)
maxLevel = 2
maxCorners = 30
qualityLevel = 0.3
minDistance = 7
blockSize = 7



