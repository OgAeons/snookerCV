import cv2
import  numpy as np
from ultralytics import YOLO
from collections import deque

# load the trained YOLO model
model = YOLO('model/v4_snookerCV.pt')

# define snooker balls color, trails
ball_color = (0, 255, 0)        # green rectangles around the balls
trail_color = (255, 255, 255)   # white tails
trail_length = 15               # max trail length
max_distance = 100              # max movement distance between balls to avoid ghost trails

# define class names 
class_names = {i: f"Snooker Ball {i}" for i in range(17)}

# dictionary to store ball trails
ball_trails = {i: deque(maxlen=trail_length) for i in range(17)}

# dictionary to store last known position of each ball
ball_positions = {}


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

        detected_ids = set()     # store detected class IDs from current frame
        new_positions = {}       # temp dictionary for updating positions

        for _, row in detections.iterrows():
            class_id = int(row['class'])          # class id
            confidence = row['confidence']        # confidence 

            # ignore detections that are not snooker balls (0-16)
            if class_id > 16:
                continue  

            # calculate box coordinates
            box = row['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

            # ignore low-confidence detections
            if confidence < 0.5:
                continue

            # calculate center of each ball
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            new_positions[class_id] = (center_x, center_y)

            # check if the ball was previously detected 
            if class_id in ball_positions:
                prev_x, prev_y = ball_positions[class_id]

                # check movement distance to prevent ghost trails
                distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                if distance > max_distance:
                    ball_trails[class_id].clear()  # clear trail to avoid jumps

            # store center point for the trail
            ball_trails[class_id].append((center_x, center_y))
            detected_ids.add(class_id)

            # draw trails 
            for i in range(1, len(ball_trails[class_id])):
                if ball_trails[class_id][i - 1] is None or ball_trails[class_id][i] is None:
                    continue
                thickness = max(1, int(3 * (1 - i / trail_length)))           # ensure thickness is always ≥ 1
                cv2.line(frame, ball_trails[class_id][i - 1], ball_trails[class_id][i], trail_color, thickness)

            # draw rectangle boxes around the balls
            cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)

            # add label to the balls
            cv2.putText(frame, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)

        # clear trails for undetected balls 
        for ball_id in ball_trails.keys():
            if ball_id not in detected_ids:
                ball_trails[ball_id].clear()  # Clear trails if ball is missing

        # update positions for the next frame
        ball_positions.update(new_positions)

        # display the frames
        cv2.imshow("Snooker Ball Tracking", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run detection on a sample video
detect_balls("videos/sample1.mp4")
