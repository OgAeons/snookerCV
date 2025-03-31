import cv2 as cv
import numpy as np

# initialize MOG2 background subtractor
bg_subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=True)

def track_balls(video):
    cap = cv.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # apply gaussian blur to reduce noise
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        # apply background subtractor
        fg_mask = bg_subtractor.apply(blur)

        cv.imshow('Snooker Ball Tracking', fg_mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

track_balls('videos/sample4.mp4')