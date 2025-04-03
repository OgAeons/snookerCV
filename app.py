import cv2
from ultralytics import YOLO

# load the trained YOLO model
model = YOLO('model/v4_snookerCV.pt')

# define color for snooker balls
ball_color = (0, 255, 0)        # Green rectangles around the balls

# define class names 
class_names = {i: f"Snooker Ball {i}" for i in range(17)}

def detect_balls(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # run YOLO detection on the frame
        results = model(frame)

        # convert results to dataframe
        detections = results[0].to_df()

        for _, row in detections.iterrows():
            class_id = int(row['class'])          # class id
            confidence = row['confidence']        # confidence 

            # ignore detections that are not snooker balls (0-16)
            if class_id > 17:
                continue  

            # calculate box coordinates
            box = row['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

            # ignore low-confidence detections
            if confidence < 0.5:
                continue

            # current ball class name
            class_name = class_names[class_id]

            # draw rectangle boxes around the balls
            cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)

            # add label to the balls
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)

        # display the frames
        cv2.imshow("Snooker Ball Tracking", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run detection on a sample video
detect_balls("videos/sample1.mp4")
