

##why did the TRACKER fail? 


import argparse
import os

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)

    df["center_dist"] = np.hypot(df["det_cx"] - df["trk_cx"],
                                 df["det_cy"] - df["trk_cy"])

    trk_area = ((df["trk_x2"] - df["trk_x1"]) * (df["trk_y2"] - df["trk_y1"]))
    df["area_ratio"] = trk_area / df["det_area"]
    return df


def first_sustained_drop(df, threshold=0.3, hold=10):
    """Find the first frame where IoU drops below `threshold` and stays there
    for at least `hold` frames. That is drift, not a momentary wobble."""
    both = (df["det_found"] == 1) & (df["trk_ok"] == 1)
    low = both & (df["iou"] < threshold)
    run = 0
    for i, is_low in enumerate(low):
        run = run + 1 if is_low else 0
        if run >= hold:
            return int(df["frame"].iloc[i - hold + 1])
    return None


def recovered(df, start_frame, threshold=0.5):
   
    after = df[df["frame"] > start_frame]
    good = after[(after["det_found"] == 1) & (after["iou"] >= threshold)]
    return int(good["frame"].iloc[0]) if len(good) else None


def frozen_windows(df, window=30, move_px=5):
   
    out = []
    for start in range(0, len(df) - window, window):
        part = df.iloc[start:start + window]
        part = part[(part["det_found"] == 1) & (part["trk_ok"] == 1)]
        if len(part) < window // 2:
            continue
        trk_moved = np.hypot(part["trk_cx"].max() - part["trk_cx"].min(),
                             part["trk_cy"].max() - part["trk_cy"].min())
        det_moved = np.hypot(part["det_cx"].max() - part["det_cx"].min(),
                             part["det_cy"].max() - part["det_cy"].min())
        if trk_moved < move_px and det_moved > move_px * 4:
            out.append((int(part["frame"].iloc[0]),
                        int(part["frame"].iloc[-1]),
                        round(trk_moved, 1), round(det_moved, 1)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="*",
                        help="one or more logs; default: both video logs")
    args = parser.parse_args()

    # With no arguments, check whichever of the standard logs exist.
    paths = args.csv or ["results/video_a_log.csv", "results/video_b_log.csv"]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("No logs found in results/. Run detect_track.py first.")

    for path in paths:
        report(path)


def report(path):
    df = load(path)
    both = df[(df["det_found"] == 1) & (df["trk_ok"] == 1)]

    print(f"\n===== {path} =====")
    print(f"frames                    {len(df)}")
    print(f"detector found object     {df['det_found'].mean():.1%} of frames")
    print(f"tracker reported a box    {df['trk_ok'].mean():.1%} of frames")
    print(f"mean IoU (both present)   {both['iou'].mean():.3f}")
    print(f"mean centre distance      {both['center_dist'].mean():.1f} px")
    print(f"max centre distance       {both['center_dist'].max():.1f} px")
    print(f"mean tracker/detector area ratio  {both['area_ratio'].mean():.2f}")

    drop = first_sustained_drop(df)
    if drop is None:
        print("\nNo sustained IoU drop ")
    else:
        back = recovered(df, drop)
        print(f"\nIoU dropped below 0.3 for good at frame {drop}")
        if back is None:
            print("re-initialisation the tracker is dead from here on.")
        else:
            print(f"  and recovered at frame {back} "
                  f"({back - drop} frames later).")

    frozen = frozen_windows(df)
    if frozen:
        print("\nWindows where the tracker box barely moved but the object "
              "did:")
        for a, b, t, d in frozen[:6]:
            print(f"  frames {a}-{b}: tracker moved {t} px, "
                  f"detector moved {d} px")
        print("  -> the tracker locked onto the background, not the object.")



if __name__ == "__main__":
    main()