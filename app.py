import cv2
from ultralytics import YOLO

# load YOLOv11
model = YOLO('yolo11n.pt')

def detect_balls(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        # read a frame from the video
        success, frame = cap.read()

        if success:
            # persisting tracks between frames
            results = model.track(frame, persist=True)

            # visualize the results
            annotated_frame = results[0].plot()

            # display the annotated frame
            cv2.imshow("Snooker Ball Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()


detect_balls("videos/sample1.mp4")