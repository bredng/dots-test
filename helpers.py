import cv2

def calculate_ave_v(x, x0, t):
    return (x-x0)/t

def get_subimage(img, x, y, width, height):
    return img[y:y+height, x:x+width]

