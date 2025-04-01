import cv2
from ultralytics import YOLO

# Loaded the custom trained model
model = YOLO('model/snookerCV.pt')

def detect_balls(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv11 tracking on the frame, persist tracks between frames
            results = model.track(frame, persist=True, show=True, tracker="bytetrack.yaml", conf=0.4) 

            # Visualizing the results on the frame
            annotated_frame = results[0].plot()

            # Displaying the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # stop the model 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # end of video
            break

    cap.release()
    cv2.destroyAllWindows()

detect_balls("videos/sample4.mp4")
