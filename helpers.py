"""
Helper functions for other scripts
"""
import cv2
import numpy as np

def calculate_ave_v(x, x0, t):
    """
    Calculate the average velocity based on SUVAT equation. 
    Must be linear movement.

    :param x: Final position
    :param x0: Initial position
    :param t: Time taken to move from initial to final position
    :return: Average velocity
    """
    return (x-x0)/t

def get_subimage(img, x, y, width, height):
    """
    Crop a frame and retrieve a subimage of a larger frame.

    :param img: The larger frame that is being cropped
    :param x: The x-coordinate of the top-left point of the cropped image
    :param y: The y-coordinate of the top-left point of the cropped image
    :param width: The width of the cropped image
    :param height: The height of the cropped image
    :return: The cropped image
    """
    return img[y:y+height, x:x+width]

def draw_box(frame, bbox, font) -> None:
    """
    Draw the box specified by the bounding box on the frame, and state that the bbox has been successfully tracked.

    :param frame: The frame that the bounding box is being drawn on
    :param bbox: The bounding box to be added to the frame
    :param font: The font being used to state that the bbox has been tracked and drawn
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(frame, "Tracking", (100, 110), font, 0.7, (0, 255, 0), 2)

def get_centre(bbox):
    """
    Calculates and returns the centre coordinates/centroid of a bounding box.

    :param bbox: The bounding box
    :return: Coordinates of the centroid of the bounding box in the form (x, y)
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    Cx = x+w/2
    Cy = y+h/2
    return (int(Cx), int(Cy))

def calculate_vector(p0, p1):
    """
    Calculates a vector between 2 pixels.

    :param p0: The coordinate of the first pixel in form (x0, y0)
    :param p1: The coordinate of the second pixel in form (x1, y1)
    :return: The magnitude and angle in degrees of the vector
    """
    Vx = p1[0] - p0[0]
    Vy = p0[1] - p1[1]
    magnitude = np.sqrt(Vx**2 + Vy**2)
    angle_r = np.arctan2(Vy, Vx)
    angle_d = angle_r*(180/np.pi)
    if angle_d < 0:
        angle_d += 360
    return magnitude, angle_d
    
def create_tracker(tracker_type: str):
    """
    Initialises a new tracker based on the input string.

    :param tracker_type: A string specifying what tracker needs to be created.
    :return: The desired tracker object 
    """
    if tracker_type.lower() == "boosting":
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type.lower() == "mil":
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type.lower() == "kcf":
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type.lower() == "tld":
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type.lower() == "medianflow":
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type.lower() == "csrt":
        return cv2.legacy.TrackerCSRT_create()
    elif tracker_type.lower() == "mosse":
        return cv2.legacy.TrackerMOSSE_create()
    else:
        print("Invalid tracker type, tracker could not be created")
