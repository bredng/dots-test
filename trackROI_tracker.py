"""
Tracks an object by selecting a region of interest with OpenCV trackers
"""
import cv2
import numpy as np

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
# TRACKER INITIALIZATION
success, frame = cap.read()
bbox = cv2.selectROI("Tracking",frame, False)
tracker.init(frame, bbox)


def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)

    if success:
        drawBox(img,bbox)
    else:
        cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(img,(15,15),(200,90),(255,0,255),2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2);
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if fps>60: myColor = (20,230,20)
    elif fps>20: myColor = (230,20,20)
    else: myColor = (20,20,230)
    cv2.putText(img,str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv2.imshow("Result", img)
    if cv2.waitKey(30) & 0xff == ord('q'):
       break
