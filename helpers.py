"""
Helper functions for other scripts
"""
import cv2
import numpy as np

# Function to calculate average velocity using initial position, displacement, and time
def calculate_ave_v(x, x0, t):
    return (x-x0)/t

# Function to get the subimage of a given image
def get_subimage(img, x, y, width, height):
    return img[y:y+height, x:x+width]

# Function to draw box on a frame
def draw_box(frame, bbox, font):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(frame, "Tracking", (100, 75), font, 0.7, (0, 255, 0), 2)

# Function to calculate the centroid of a bounding box
def get_centre(bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    Cx = x+w/2
    Cy = y+h/2
    return (int(Cx), int(Cy))

# Function to calculate the vector of 2 points
def calculate_vector(p0, p1):
    Vx = p1[0] - p0[0]
    Vy = p0[1] - p1[1]
    magnitude = np.sqrt(Vx**2 + Vy**2)
    angle_r = np.arctan2(Vy, Vx)
    angle_d = angle_r*(180/np.pi)
    return magnitude, angle_d
    
