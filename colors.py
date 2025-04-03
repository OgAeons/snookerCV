import cv2
import numpy as np

def get_color(color):
    # hsv color ranges
    color_ranges = {
        "red": [(0, 150, 100), (5, 255, 255), (175, 150, 100), (180, 255, 255)],
        "orange": [(11, 150, 100), (20, 255, 255)],
        "blue": [(86, 50, 50), (130, 255, 255)],  
        "green": [(36, 100, 100), (80, 255, 255)], 
        "yellow": [(25, 100, 100), (45, 255, 255)], 
        "black": [(0, 0, 0), (180, 255, 50)],
        "white": [(35, 20, 230), (50, 60, 255)]
    }
    return color_ranges.get(color.lower(), None)


def generate_mask(frame, selected_color):
    range = get_color(selected_color)
    if not range:
        return None

    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    if len(range) == 4:
        mask1 = cv2.inRange(hsv, np.array(range[:2][0]), np.array(range[:2][1]))
        mask2 = cv2.inRange(hsv, np.array(range[2:4][0]), np.array(range[2:4][1]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, np.array(range[0]), np.array(range[1]))

    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask_blurred