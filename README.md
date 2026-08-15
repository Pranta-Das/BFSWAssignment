# BFSW Assignment: Object Detection, Tracking and Failure Analysis

## What this does

A fixed RGB camera watches a workspace. The pipeline finds a target object,
tracks it, and reports its pixel position (u, v) each frame.



## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
python detect_track.py --video data/video_a.mp4
python detect_track.py --video data/video_b.mp4

python analyze.py




## tracker failures
python diagnose_tracker.py

## detector failure
python verify.py --frame 190 --good-frame 100

## re-init experiment
python detect_track.py --video data/video_b.mp4 --reinit-on-lost
python diagnose_tracker.py --csv results/video_b_reinit_log.csv

## screenshot

python analyze.py --frames 190,700,435




```

## Files


(detect_track.py) detection + tracking, writes the per-frame CSV log 

(analyze.py) summary numbers, failure windows, comparison figure

(verify.py)  tests the cause of a specific failure

(diagnose_traker.py)  check tracker faliures

(experiment_log.md)  what was tried, what failed, and why


## Videos

input video: 

video-a: https://drive.google.com/file/d/1_2-e4IsE1nwhKVP4PzxZBCTJ046hXSzz/view?usp=sharing

video-b: https://drive.google.com/file/d/1Dz4d5mhYJc603CLw4vCyj79nRU1i2RMo/view?usp=sharing

Result: 

video-a-out: https://drive.google.com/file/d/13UvVve8QmXMpZiskNroPtDbOLC-hziKr/view?usp=sharing

video-b-out: https://drive.google.com/file/d/1hSmYD3lR_jMZ5v9VO21dAO4BnOV5cqM9/view?usp=sharing
