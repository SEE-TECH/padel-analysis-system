# Padel Match Analysis System

A computer vision system for analyzing padel matches using deep learning. The system detects and tracks players and the ball, analyzes court positions, and generates detailed match statistics.

## Features

- **Player Detection & Tracking**: Detects and tracks all 4 players on the padel court using YOLOv8
- **Ball Detection & Tracking**: Tracks the ball using TrackNet architecture with trajectory interpolation
- **Court Detection**: Automatic padel court line detection for position mapping
- **Shot Analysis**: Detects shots, calculates ball speed, and identifies shot types
- **Mini Court Visualization**: Bird's-eye view showing player positions and ball movement
- **Player Statistics**: Shot count, movement speed, and performance metrics per player
- **Multi-Video Analysis**: Process multiple match videos and aggregate statistics
- **Player Highlights**: Generate highlight reels for specific players

## Project Structure

```
padel-analysis-system/
├── main.py                      # Main entry point
├── multi_video_analysis.py      # Multi-video processing
├── player_highlights.py         # Generate player highlight videos
├── models/
│   ├── tracknet.py              # TrackNet ball detection model
│   └── weights/                 # Model weights (not included)
├── trackers/
│   ├── player_tracker.py        # YOLOv8 player tracking
│   ├── ball_tracker.py          # YOLO ball tracking
│   └── tracknet_ball_tracker.py # TrackNet ball tracking
├── court_line_detector/
│   └── court_line_detector.py   # Court keypoint detection
├── mini_court/
│   └── mini_court.py            # Mini court visualization
├── visualizations/
│   └── ball_visualizer.py       # Ball trajectory visualization
├── utils/
│   ├── video_utils.py           # Video I/O
│   ├── bbox_utils.py            # Bounding box utilities
│   ├── conversions.py           # Coordinate conversions
│   └── player_stats_drawer_utils.py
├── inputs/                      # Input videos
├── output_videos/               # Processed videos
└── tracker_stubs/               # Cached detections
```

## Installation

```bash
git clone https://github.com/SEE-TECH/padel-analysis-system.git
cd padel-analysis-system
pip install -r requirements.txt
```

## Usage

### Single Video Analysis

```bash
python main.py
```

### Multi-Video Analysis

```bash
python multi_video_analysis.py
```

### Generate Player Highlights

```bash
python player_highlights.py
```

## Model Weights

Model weights are not included in this repository due to file size. Place the following in the `models/` directory:

- `yolov8x.pt` - Player detection
- `best.pt` - Ball detection (YOLO)
- `keypoints_model.pth` - Court keypoint detection
- `weights/ball_detection/TrackNet_best.pt` - TrackNet ball detection

## Output

The system generates annotated videos with:
- Player bounding boxes with IDs
- Ball position and trajectory
- Mini court with real-time positions
- Player statistics overlay
- Shot detection markers
