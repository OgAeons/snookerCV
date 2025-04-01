import cv2
from ultralytics import YOLO

# Load the custom trained model
model = YOLO('model/snookerCV.pt')

def detect_balls(video):
    cap = cv2.VideoCapture(video)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # results from yolov11
            results = model(frame)

            # Convert results to to_df data
            detections = results[0].to_df()  

            for index, row in detections.iterrows():
                # bounding box coordinates 
                box = row['box']
                x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

                # class index ie.0
                ball_id = int(row['class'])  

                # Draw bounding box around detected ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label the ball with its ID
                cv2.putText(frame, f'Snooker Ball {ball_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # display the bounding boxes and labels on the video
            cv2.imshow("Snooker Ball Detection", frame)

            # stop the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# function call and path to video sample
detect_balls("videos/sample4.mp4")
