# Smart Office Person Tracking with VLM

AI-powered smart office monitoring system that performs:

-Real-time person detection

-Multi-zone tracking (Reception, Office, Meeting Room)

-DeepSORT ID tracking

-Clothing color recognition

-Behavior inference (Anxious, Calm, Rushing, Confused, etc.)

-Movement event logging to CSV

Built using:

-Python

-YOLOv8

-DeepSORT

-Shapely (zone detection)

-OpenCV

# Features

**1.Person Detection**

*Uses YOLOv8n for real-time human detection.

*Configurable confidence threshold

*Optimized image size

*Person class filtering only

**2.Multi-Object Tracking**

*Powered by DeepSORT:

*Unique ID assignment

*Track persistence

*Motion history tracking

*Re-identification support

**3.Smart Zone Monitoring**

The office is divided into 3 zones:

-Zone Area

-Reception	Left section

-Office	Center section

-Meeting Room	Right section

The system detects:

-Entry

-Exit

-Zone transitions

-Unusual movement patterns

-Stationary behavior

Zone changes are logged with timestamps.

**4.Clothing Color Recognition**

-Extracts upper-body region

-LAB color space masking

-CSS3 color matching (if webcolors available)

-Fallback custom color classification

Detects:

-Red

-Blue

-Green

-Black

-White

-Beige

-Brown

-Pink

-Purple

-Gray variations

-And more

Used for:

-Profile identification

-Human-readable tracking logs

**5.Behavior Analysis Engine**

Based on:

-Speed

-Motion variance

-Direction change (zig-zag detection)

-Stability time

-Movement patterns

Possible behavior classifications:

-Standing

-Walking (Left / Right / Up / Down)

-Calm / Relaxed

-Curious / Exploring

-Focused / Determined

-Rushing / Hurried

-Anxious / Nervous

-Angry / Agitated

-Confused / Lost

-Sad / Tired

-Cautious / Careful

-Happy / Energetic

**6.Event Logging**

Every zone change generates a structured event:

[2026-02-22 10:42:13] ID 3 MOVED Reception -> Office

Saved to:movement_events.csv

CSV includes:

-ID

-Clothing

-Behavior

-Zone

-Zone Status

-Visited Zones

-Timestamp

# System Architecture

YOLOv8 → DeepSORT → Zone Detection → Behavior Analysis → Event Logger → Visualization

# Installation

**1️)Clone Repository**

git clone https://github.com/your-username/smart-office-vlm.git

cd smart-office-vlm

**2️)Install Dependencies**

pip install -r requirements.txt

Or manually:

pip install ultralytics opencv-python numpy pandas shapely deep-sort-realtime torch torchvision webcolors

▶️ Run the System

If using camera:

python sop12.py

If using video file:

Update inside config:

CFG["video"] = "your_video.mp4"

# Configuration

Inside CFG:

CFG = {

    "model": "yolov8n.pt",
    
    "video": "test_office_video.mp4",
    
    "w": 1280,
    
    "h": 720,
    
    "sz": 640,
    
    "conf": 0.45,
    
    "iou": 0.5,
    
    "csv": "movement_events.csv"
}

You can modify:

-Model size

-Resolution

-Confidence threshold

# Use Cases

-Smart office analytics

-Workplace behavior monitoring

-Security zone monitoring

-Retail analytics

-Smart building automation

-Research in behavior inference

# Future Improvements

-Face embedding (privacy-safe vector matching)

-VLM-based semantic reasoning

-Web dashboard

-Heatmap visualization

-Multi-camera support

-Edge device deployment

-REST API integration

-Output CSV name
