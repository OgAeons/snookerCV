import cv2
import argparse
import  numpy as np
from ultralytics import YOLO
from collections import deque
from colors import generate_mask

# load the trained YOLO model
model = YOLO('model/v4_snookerCV.pt')

# color parameters
ball_color = (0, 255, 0)        # green box around the balls
trail_color = (255, 255, 255)   # white tails

# tracking parameters
trail_length = 15               # max trail length
max_distance = 50               # max movement distance between balls to avoid ghost trails
min_mask_per = 0.5             # min percentage of pixels in the mask for detection

# tracking storage
ball_trails =  {}
ball_positions = {}
ball_ids = {}
next_ball_id = 0


# assign a unique id to each ball
def set_ball_id(center):
    global next_ball_id
    closest_id = None
    min_dist = max_distance

    for ball_id, prev_center in ball_positions.items():
        dist = np.linalg.norm(np.array(center) - np.array(prev_center))
        if dist < min_dist:
            min_dist = dist
            closest_id = ball_id

    # returns closest ball id if center - prevcenter < max distance
    if closest_id is not None:
        return closest_id
    
    # returns new id if center - prevcenter not < max distance
    new_id = next_ball_id
    next_ball_id+=1
    return new_id


def detect_balls(video, selected_color):
    global ball_positions

    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # run YOLO detection on the frame
        results = model(frame)

        # convert results to dataframe
        detections = results[0].to_df()

        # generate hsv mask
        mask = generate_mask(frame, selected_color)
        if mask is None:
            print("invalid color")
            break

        new_positions = {}       # temp dictionary for updating positions

        for _, row in detections.iterrows():
            # ignore if not class (0-16) & confiednce below 0.5
            if int(row['class']) > 16 or row['confidence'] < 0.5:
                continue

            # calculate box coordinates
            x1, y1, x2, y2 = map(int, [row['box']['x1'], row['box']['y1'], row['box']['x2'],row['box']['y2']])

            # calculate center of each ball
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # check if ball is in color range
            mask_range = mask[y1:y2, x1:x2]
            mask_area = cv2.countNonZero(mask_range)
            box_area = (x2 - x1) * (y2 - y1)

            if box_area > 0 and mask_area / box_area >= min_mask_per:
                # set unique id for balls
                ball_id = set_ball_id(center)
                new_positions[ball_id] = center

                # draw boxes around the balls
                cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
                cv2.putText(frame, f"Snooker Ball {ball_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)

                # check if ball id exists in ball_trails
                if ball_id not in ball_trails:
                    ball_trails[ball_id] = deque(maxlen=trail_length)

                ball_trails[ball_id].append(center)

                # draw trails for the balls
                for i in range(1, len(ball_trails[ball_id])):
                    if ball_trails[ball_id][i-1] and ball_trails[ball_id][i]:
                        cv2.line(frame, ball_trails[ball_id][i-1], ball_trails[ball_id][i], trail_color, 3)

        # update positions for the next frame
        ball_positions = new_positions

        # display the frames
        cv2.imshow("Snooker Ball Tracking", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snooker Ball Tracking")
    parser.add_argument("color", type=str, help="Color to track [ red, orange, blue, green, yellow, black ]")
    args = parser.parse_args()
    
    selected_color = args.color.lower()
    detect_balls("videos/sample2.mp4", selected_color)
