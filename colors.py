import cv2
import numpy as np

def get_color(color):
    # hsv color ranges
    color_ranges = {
        "red": [(0, 150, 100), (5, 255, 255), (175, 150, 150), (180, 255, 255)],
        "orange": [(11, 150, 100), (20, 255, 255)],
        "blue": [(215, 150, 100), (240, 255, 255)],
        "green": [(120, 150, 100), (150, 255, 255)],
        "yellow": [(50, 150, 100), (65, 255, 255)],
        "pink": [(330, 100, 100), (345, 255, 255)],
        "brown": [(15, 100, 50), (30, 255, 150)],
        "black": [(0, 0, 0), (180, 255, 50)],
        "white": [(0, 0, 200), (180, 30, 255)],
    }

    return color_ranges.get(color.lower(), None)


def generate_mask(frame, selected_color):
    # check color 
    range = get_color(selected_color)
    if not range:
        return None
    
    # convert the frame to hsv color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # handle the split hues of hsv color wheel
    if len(range) == 4:
        mask1 = cv2.inRange(hsv, np.array(range[:2][0]), np.array(range[:2][1]))
        mask2 = cv2.inRange(hsv, np.array(range[2:4][0]), np.array(range[2:4][1]))
        return cv2.bitwise_or(mask1, mask2)
    
    return cv2.inRange(hsv, np.array(range[0]), np.array(range[1]))

