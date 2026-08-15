## why did the DETECTOR fail

import argparse
import os

import cv2
import numpy as np


from detect_track import Detector


def read_frame(video, number):
    """Read one specific frame from a video."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read frame {number} from {video}")
    return frame


# each function tests one possible cause 

def fix_contrast(frame):
    """Test: was the problem low local contrast? (CLAHE evens it out.)"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def fix_brightness(frame, gamma=0.5):
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype("uint8")
    return cv2.LUT(frame, table)


def fix_size(frame, factor=2.0):
    return cv2.resize(frame, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def fix_blur(frame):
   
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.6, blurred, -0.6, 0)


def measure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var(), gray.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True,
                        help="the failing frame number")
    parser.add_argument("--video", default="data/video_b.mp4")
    parser.add_argument("--good-frame", type=int,
                        help="a working frame, to compare against")
    parser.add_argument("--target", default="cup")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="lower than the pipeline threshold, so we can "
                             "see confidence that only just fell below it")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = Detector(args.weights, args.target, args.conf)
    frame = read_frame(args.video, args.frame)

    sharp, bright = measure(frame)
    print(f"\nFailing frame {args.frame}: sharpness {sharp:.1f}, "
          f"brightness {bright:.1f}")

    if args.good_frame is not None:
        good_sharp, good_bright = measure(read_frame(args.video,
                                                     args.good_frame))
        print(f"Working frame {args.good_frame}: sharpness {good_sharp:.1f}, "
              f"brightness {good_bright:.1f}")

    tests = [
        ("original (no change)", frame),
        ("contrast fixed", fix_contrast(frame)),
        ("brightened", fix_brightness(frame, 0.5)),
        ("darkened", fix_brightness(frame, 1.8)),
        ("enlarged 2x", fix_size(frame, 2.0)),
        ("sharpened", fix_blur(frame)),
    ]

    print(f"\n{'test':<22} {'confidence':>11} {'change':>9}")
    print("-" * 45)

    baseline = None
    for name, image in tests:
        _, conf = detector.detect(image)
        if baseline is None:
            baseline = conf
        print(f"{name:<22} {conf:>11.3f} {conf - baseline:>+9.3f}")

        out = os.path.join(
            args.out_dir,
            f"verify_f{args.frame}_{name.split()[0]}.jpg")
        cv2.imwrite(out, image)



if __name__ == "__main__":
    main()