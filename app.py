import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('model/snookerCv_v3.5.pt')

# Define color for snooker balls
BALL_COLOR = (0, 255, 0)     # Green 

# Define class names 
CLASS_NAMES = {i: f"Snooker Ball {i}" for i in range(17)}

def detect_balls(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection on the frame
        results = model(frame)

        # convert to dataframe
        detections = results[0].to_df()

        for _, row in detections.iterrows():
            class_id = int(row['class'])
            confidence = row['confidence']

            # Ignore detections that are not snooker balls (0-15)
            if class_id > 17:
                continue  

            # Extract bounding box coordinates
            box = row['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

            # ignore low-confidence detections
            if confidence < 0.4:
                continue

            # Get class name
            class_name = CLASS_NAMES[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)

            # Add label
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)

        # Show frame
        cv2.imshow("Snooker Ball Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run detection on a sample video
detect_balls("videos/sample4.mp4")
