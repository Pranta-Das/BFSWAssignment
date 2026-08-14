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
python detect_track.py --video data/video_a.mp4 --target cup \
    --out-video results/video_a_out.mp4 --out-csv results/video_a_log.csv

python detect_track.py --video data/video_b.mp4 --target cup \
    --out-video results/video_b_out.mp4 --out-csv results/video_b_log.csv

python analyze.py --csv-a results/video_a_log.csv \
    --csv-b results/video_b_log.csv --out-dir results

python verify.py --video data/video_b.mp4 --frame 515 --good-frame 120 \
    --target cup
```

## Files


(detect_track.py) detection + tracking, writes the per-frame CSV log 

(analyze.py) summary numbers, failure windows, comparison figure

(verify.py)  tests the cause of a specific failure

(check_videos.py)  sanity-checks recordings before processing

(experiment_log.md)  what was tried, what failed, and why


## Videos

Check thhe result folder.


