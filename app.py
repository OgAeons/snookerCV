import cv2 as cv
import numpy as np

def track_balls(video):
    cap = cv.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('Snooker Ball Tracking', gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

track_balls('videos/sample1.mp4')