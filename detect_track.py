"""
detect_track.py
---------------
Runs a detector and a tracker over a video AT THE SAME TIME, in two
independent branches, and writes one row of measurements per frame to a CSV.

Why two independent branches instead of "detect once, then track"?
Because it lets us tell WHICH part failed. If the detector confidence stays
high but the two boxes stop overlapping, the tracker drifted. If the tracker
still reports a box but the detector found nothing, the object is still
roughly where we think it is but the detector can no longer recognise it.

Run:
    python detect_track.py --video data/video_a.mp4 --target cup \
        --out-video results/video_a_out.mp4 --out-csv results/video_a_log.csv
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Detector
# ---------------------------------------------------------------------------
class Detector:
    """Wraps a pretrained YOLO model and returns only the target object."""

    def __init__(self, weights, target, conf):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf

        # YOLO knows 80 COCO classes. Find the numeric id of the one we want,
        # so the model can ignore everything else.
        self.class_ids = [
            i for i, name in self.model.names.items()
            if name.lower() == target.lower()
        ]
        if not self.class_ids:
            raise ValueError(
                f"'{target}' is not a class this model knows.\n"
                f"Valid names: {sorted(self.model.names.values())}"
            )

    def detect(self, frame):
        """Return (box, confidence). box is [x1, y1, x2, y2], or None."""
        result = self.model.predict(
            frame, conf=self.conf, classes=self.class_ids, verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None, 0.0

        # If several objects are found, keep only the most confident one.
        confs = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        box = result.boxes.xyxy.cpu().numpy()[best].astype(float)
        return box, float(confs[best])


# ---------------------------------------------------------------------------
# 2. Tracker
# ---------------------------------------------------------------------------
def make_tracker(name="CSRT"):
    """Create an OpenCV tracker.

    CSRT only exists in opencv-contrib-python, and OpenCV 5.0 removed it
    completely. So we try several spellings, and fall back to MIL (which is
    in every build) with a loud warning rather than crashing.
    """
    options = {
        "CSRT": ["TrackerCSRT.create", "TrackerCSRT_create",
                 "legacy.TrackerCSRT_create"],
        "KCF":  ["TrackerKCF.create", "TrackerKCF_create",
                 "legacy.TrackerKCF_create"],
        "MIL":  ["TrackerMIL.create", "TrackerMIL_create"],
    }

    # Try the tracker we asked for first, then the others as a fallback.
    order = [name.upper()] + [n for n in ("CSRT", "KCF", "MIL")
                              if n != name.upper()]

    for want in order:
        for path in options.get(want, []):
            obj = cv2
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                tracker = obj()
            except (AttributeError, TypeError, cv2.error):
                continue

            if want != name.upper():
                print(f"[WARNING] {name} is not available in cv2 "
                      f"{cv2.__version__}. Using {want} instead. "
                      f"This changes the tracking results -- write it in "
                      f"experiment_log.md.")
            return tracker

    raise RuntimeError(
        f"No tracker available in cv2 {cv2.__version__}. "
        f"Install opencv-contrib-python<5"
    )


# ---------------------------------------------------------------------------
# 3. Small helper functions
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    """Intersection over Union: how much two boxes overlap. 0 = not at all."""
    if box_a is None or box_b is None:
        return 0.0

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - overlap

    return float(overlap / union) if union > 0 else 0.0


def center(box):
    """Middle point (u, v) of a box -- this is what a robot would aim at."""
    if box is None:
        return None, None
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def to_xywh(box):
    """OpenCV trackers want (x, y, width, height)."""
    return (int(box[0]), int(box[1]),
            int(box[2] - box[0]), int(box[3] - box[1]))


def to_xyxy(box):
    """Convert back from (x, y, w, h) to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def image_quality(frame):
    """Measure the frame itself, not the detection.

    These two numbers are the physical evidence for the failure analysis.
    Without them you can only say "the detector failed"; with them you can
    say "the detector failed while sharpness dropped from 340 to 42".
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
    }


def get_status(det_box, trk_box, iou_value):
    """One word describing what the two branches did this frame."""
    if det_box is None and trk_box is None:
        return "BOTH_LOST"
    if det_box is not None and trk_box is None:
        return "DET_ONLY"      # tracker lost it
    if det_box is None and trk_box is not None:
        return "TRK_ONLY"      # detector lost it
    return "AGREE" if iou_value >= 0.3 else "DISAGREE"


def draw_overlay(frame, frame_id, det_box, det_conf, trk_box, iou_value,
                 status, quality):
    """Draw both boxes and the measurements onto a copy of the frame."""
    out = frame.copy()
    green, orange = (0, 220, 0), (0, 165, 255)

    # Detector box in green, tracker box in orange.
    for box, color, label in ((det_box, green, "det"), (trk_box, orange, "trk")):
        if box is None:
            continue
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        cv2.rectangle(out, p1, p2, color, 2)

        cx, cy = center(box)
        cv2.circle(out, (int(cx), int(cy)), 5, color, -1)

        text = f"{label} {det_conf:.2f}" if label == "det" else label
        cv2.putText(out, text, (p1[0], max(p1[1] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show the pixel coordinate of the detector centre -- this is the
        # (u, v) that the position-estimation stage would receive.
        if label == "det":
            cv2.putText(out, f"(u={cx:.0f}, v={cy:.0f})", (p1[0], p2[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    lines = [
        f"frame {frame_id}   {status}",
        f"conf {det_conf:.2f}   iou {iou_value:.2f}",
        f"sharp {quality['sharpness']:.0f}   bright {quality['brightness']:.0f}",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 26
        # Black outline first, then white text, so it is readable on any frame.
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
    return out


# Column names of the CSV log.
COLUMNS = [
    "frame", "time_s", "status",
    "det_found", "det_conf", "det_cx", "det_cy", "det_area",
    "det_x1", "det_y1", "det_x2", "det_y2",
    "trk_ok", "trk_cx", "trk_cy",
    "trk_x1", "trk_y1", "trk_x2", "trk_y2",
    "iou", "det_ms", "sharpness", "brightness",
]


# ---------------------------------------------------------------------------
# 4. Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--target", default="cup", help="COCO class name")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="CSRT")
    parser.add_argument("--reinit-on-lost", action="store_true",
                        help="Restart the tracker from the detector when it "
                             "fails. OFF by default so drift stays visible.")
    args = parser.parse_args()

    # Make sure the output folders exist.
    os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    detector = Detector(args.weights, args.target, args.conf)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.out_video,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    csv_file = open(args.out_csv, "w", newline="")
    log = csv.DictWriter(csv_file, fieldnames=COLUMNS)
    log.writeheader()

    tracker = None
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break   # end of video

        # --- branch 1: detector ------------------------------------------
        start = time.perf_counter()
        det_box, det_conf = detector.detect(frame)
        det_ms = (time.perf_counter() - start) * 1000.0

        # --- branch 2: tracker (does NOT look at the detector) ------------
        trk_box = None
        if tracker is None:
            # Start the tracker once, from the first detection we get.
            if det_box is not None:
                tracker = make_tracker(args.tracker)
                tracker.init(frame, to_xywh(det_box))
                trk_box = det_box.copy()
        else:
            tracker_ok, box = tracker.update(frame)
            if tracker_ok:
                trk_box = to_xyxy(box)
            elif args.reinit_on_lost and det_box is not None:
                tracker = make_tracker(args.tracker)
                tracker.init(frame, to_xywh(det_box))
                trk_box = det_box.copy()

        # --- compare the two branches ------------------------------------
        iou_value = iou(det_box, trk_box)
        status = get_status(det_box, trk_box, iou_value)
        quality = image_quality(frame)

        det_cx, det_cy = center(det_box)
        trk_cx, trk_cy = center(trk_box)

        def rnd(value, digits=2):
            """Round, but keep None as None so the CSV cell stays empty."""
            return None if value is None else round(value, digits)

        log.writerow({
            "frame": frame_id,
            "time_s": round(frame_id / fps, 3),
            "status": status,
            "det_found": int(det_box is not None),
            "det_conf": round(det_conf, 4),
            "det_cx": rnd(det_cx), "det_cy": rnd(det_cy),
            "det_area": None if det_box is None else round(
                (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]), 1),
            "det_x1": rnd(None if det_box is None else det_box[0]),
            "det_y1": rnd(None if det_box is None else det_box[1]),
            "det_x2": rnd(None if det_box is None else det_box[2]),
            "det_y2": rnd(None if det_box is None else det_box[3]),
            "trk_ok": int(trk_box is not None),
            "trk_cx": rnd(trk_cx), "trk_cy": rnd(trk_cy),
            "trk_x1": rnd(None if trk_box is None else trk_box[0]),
            "trk_y1": rnd(None if trk_box is None else trk_box[1]),
            "trk_x2": rnd(None if trk_box is None else trk_box[2]),
            "trk_y2": rnd(None if trk_box is None else trk_box[3]),
            "iou": round(iou_value, 4),
            "det_ms": round(det_ms, 2),
            "sharpness": round(quality["sharpness"], 2),
            "brightness": round(quality["brightness"], 2),
        })

        writer.write(draw_overlay(frame, frame_id, det_box, det_conf,
                                  trk_box, iou_value, status, quality))

        frame_id += 1
        if frame_id % 50 == 0:
            print(f"  processed {frame_id} frames...")

    cap.release()
    writer.release()
    csv_file.close()
    print(f"Done. {frame_id} frames -> {args.out_video} and {args.out_csv}")


if __name__ == "__main__":
    main()
