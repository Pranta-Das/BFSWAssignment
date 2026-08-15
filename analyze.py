## Analyze the tracking results

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
import pandas as pd


def summary(name, df):
    """Print the numbers you will quote in the report."""
    healthy = df[df["status"] == "AGREE"]
    found = df[df["det_found"] == 1]

    print(f"\n===== Video {name} =====")
    print(f"frames                 {len(df)}")
    print(df["status"].value_counts().to_string())
    print(f"detection rate         {df['det_found'].mean():.1%}")
    if len(found):
        print(f"mean confidence        {found['det_conf'].mean():.3f}  "
              f"(only frames with a detection)")
    else:
        print("mean confidence        -- (the object was never detected; "
              "check --target and --conf)")
    if len(healthy):
        print(f"healthy baseline conf  {healthy['det_conf'].mean():.3f}  ")
    print(f"brightness             {df['brightness'].min():.0f} to "
          f"{df['brightness'].max():.0f}")
    print(f"mean detector time     {df['det_ms'].mean():.1f} ms "
          f"({1000 / df['det_ms'].mean():.1f} FPS)")


def find_failures(df, min_length=3):
    """Group runs of consecutive non-AGREE frames into failure windows."""
    is_bad = (df["status"] != "AGREE").tolist()

    windows = []
    start = None
    for i, bad in enumerate(is_bad):
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            windows.append((start, i - 1))
            start = None
    if start is not None:
        windows.append((start, len(df) - 1))

    rows = []
    for a, b in windows:
        if b - a + 1 < min_length:
            continue                      # ignore 1-2 frame blips
        part = df.iloc[a:b + 1]
        rows.append({
            "start": int(part["frame"].iloc[0]),
            "end": int(part["frame"].iloc[-1]),
            "frames": b - a + 1,
            "type": ",".join(sorted(part["status"].unique())),
            "conf": round(part["det_conf"].mean(), 3),
            "sharpness": round(part["sharpness"].mean(), 1),
            "brightness": round(part["brightness"].mean(), 1),
        })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("frames", ascending=False)
    return result


def make_plot(df_a, df_b, out_dir):
    """One figure, four stacked panels, both videos on the same axes."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    panels = [
        ("det_conf", "detector confidence", (0, 1.05)),
        ("iou", "IoU detector vs tracker", (0, 1.05)),
        ("brightness", "brightness", None),
        ("sharpness", "sharpness", None),
    ]

    for ax, (column, label, ylim) in zip(axes, panels):
        for df, name, color in ((df_a, "A - normal", "#2c7fb8"),
                                (df_b, "B - challenging", "#d95f0e")):
            if df is None:
                continue
            values = df[column]
           
            if column == "iou":
                values = values.where(df["det_found"] == 1)
            ax.plot(df["frame"], values, linewidth=1.1, label=name, color=color)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0].legend(loc="lower left", fontsize=9)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Per-frame evidence: normal vs challenging condition")
    fig.tight_layout()

    path = os.path.join(out_dir, "evidence_curves.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


def save_frames(video, frame_numbers, out_dir):
    """Pull specific frames out of the annotated output video."""
    cap = cv2.VideoCapture(video)
    for n in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, image = cap.read()
        if not ok:
            print(f"could not read frame {n}")
            continue
        path = os.path.join(out_dir, f"failure_frame_{n:05d}.png")
        cv2.imwrite(path, image)
        print(f"saved {path}")
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-a", default="results/video_a_log.csv")
    parser.add_argument("--csv-b", default="results/video_b_log.csv")
    parser.add_argument("--video", help="annotated video to cut frames from; "
                                        "default: results/video_b_out.mp4")
    parser.add_argument("--frames", help="e.g. 190,700,435")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    csv_a = args.csv_a if os.path.exists(args.csv_a) else None
    csv_b = args.csv_b if os.path.exists(args.csv_b) else None
    if csv_a is None and csv_b is None:
        raise SystemExit(f"No logs found. Expected {args.csv_a} or "
                         f"{args.csv_b}. Run detect_track.py first.")

    # Frames are cut from video B's annotated output unless told otherwise.
    if args.frames and args.video is None:
        args.video = "results/video_b_out.mp4"

    df_a = pd.read_csv(csv_a) if csv_a else None
    df_b = pd.read_csv(csv_b) if csv_b else None

    for name, df in (("A", df_a), ("B", df_b)):
        if df is not None:
            df["det_conf"] = df["det_conf"].fillna(0.0)
            summary(name, df)

    if df_b is not None:
        failures = find_failures(df_b)
        print("\n===== Failure windows in video B (longest first) =====")
        print(failures.to_string(index=False) if len(failures) else "none")
       

    if df_a is not None or df_b is not None:
        make_plot(df_a, df_b, args.out_dir)

    if args.frames and args.video:
        save_frames(args.video,
                    [int(x) for x in args.frames.split(",")],
                    args.out_dir)


if __name__ == "__main__":
    main()