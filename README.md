# Snooker Ball Tracking - ` snookerCV `

## Overview
This project is a **computer vision-based solution** designed to track snooker balls in a video using **a custom-trained YOLOv11 model and OpenCV**. The system accurately detects, labels, and tracks snooker balls, assigning unique IDs and drawing trails to visualize movement. Additionally, it allows for tracking only specific colored balls using **color segmentation**, ensuring precise ball identification.

### **Key Features**
✅ **Custom-labeled dataset** for improved accuracy (Roboflow).  
✅ **Custom-trained YOLOv11 model** for snooker ball detection.  
✅ Detects and labels snooker balls with bounding boxes.  
✅ Assigns unique IDs to each ball for tracking.  
✅ Draws movement trails to show the ball's path.   
✅ Implements **color segmentation** for precise tracking.
✅ Filters and tracks only selected colored balls. 
✅ Supports recorded video processing.  

---

## Installation
### 1. Install Python
Ensure you have Python **3.8 or later** installed. Download it here:  
🔗 [Python Official Website](https://www.python.org/downloads/)

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/snookerCV.git
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
This installs all required packages, including:
- `ultralytics` (for YOLOv11)
- `opencv-python`
- `numpy`

---

## Model & Data Setup
### **1. YOLOv11 Model**
The trained YOLOv11 model (`v4_snookerCV.pt`) is already included in the repository under the `model/` folder. No extra downloads are required.
- `model/v4_snookerCV.pt`

### **2. Sample Test Data (Videos & Images)**
Sample test videos are included in the repository in the `videos/` folder:
- `videos/sample1.mp4`
- `videos/sample2.mp4`
- `videos/sample3.mp4`

You can use these videos for testing or provide your own video files.

---

## Run Code
This project includes **two separate scripts**:
1. `app.py` – Tracks **all balls** in the video (normal detection & tracking)
2. `color_tracker.py` – Tracks **only a selected ball color** (color-based tracking)

### **1. Run Normal Snooker Ball Tracking (All Balls)**
This script detects, labels, and tracks **all balls** with unique IDs and movement trails.
```bash
python3 app.py {video sample path}
```
*Ex:*\
Use `python` (windows) and `python3` (mac)
```bash
python3 app.py videos/sample1.mp4
``` 
or
```bash
python3 app.py videos/sample2.mp4
```


### **2. Run Color-Specific Snooker Ball Tracking**
This script detects and tracks **only the selected color ball**.
```bash
python3 color_tracker.py {color}
```
*Ex:*\
Use `python` (windows) and `python3` (mac)
```bash
python3 color_tracker.py red
```
Select a color from: `red, orange, yellow, green, blue, black`

---

## Demonstration Videos
Videos demonstrating both methods will be added in the README:
📌 **Snooker Ball Tracking** (app.py) – *[Video Link]()*  
📌 **Color-Based Ball Tracking** (color_tracker.py) – *[Video Link]()*  

---

## Custom Dataset & Licensing
This project includes a **custom dataset** labelled on **[Roboflow](https://roboflow.com/)** for improved model accuracy. The dataset was manually labeled for snooker ball detection and classification.
- **Dataset License:** The custom dataset is licensed for open use with attribution. (CC by 4.0)
- **Roboflow Dataset Link:** 🔗 [Custom Labelled Dataset](https://universe.roboflow.com/any-doggo/snookerballv3/dataset/6)  

The dataset was used to train a YOLOv11 model specifically for detecting and tracking snooker balls.

---

## References
🔹 **YOLOv11 Documentation** – [Ultralytics YOLOv11](https://docs.ultralytics.com/quickstart/)  
🔹 **OpenCV Documentation** – [OpenCV Official Docs](https://docs.opencv.org/4.x/index.html)  
🔹 **Roboflow Object Detection** – [Roboflow](https://roboflow.com/)  
🔹 **Kaggle Snooker Dataset** – [Kaggle](https://www.kaggle.com/search?q=Snooker+ball+dataset)  
🔹 **Pexels Video Samples** – [Pexels](https://www.pexels.com/) 

---