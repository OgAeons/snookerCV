import cv2
from ultralytics import YOLO

# load YOLOv11 model
model = YOLO('yolo11n.pt')

def detect_balls(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # run YOLO model on the frame
        results = model(frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # box coordinates
            conf = box.conf.item()  # confidence score
            cls = int(box.cls.item())  # class index

            # filter detections by confidence threshold
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # display the frame
        cv2.imshow("Snooker Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_balls("videos/sample1.mp4")
