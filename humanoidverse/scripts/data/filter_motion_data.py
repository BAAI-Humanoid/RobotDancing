import argparse
import glob
import os
import shutil

import joblib
import numpy as np
from tqdm import tqdm


def get_frame_count(data):
    """Return the frame count of the first motion."""
    motion_data = data[next(iter(data))]
    return motion_data["root_trans_offset"].shape[0]


def filter_motion_data(
    directory_path,
    min_frames=120,
    max_frames=150,
    keep_mode="move",
    output_dir=None,
):
    """Move or delete motion files outside the requested frame range."""
    print(f"Scanning: {directory_path}")
    print(f"Frame range: {min_frames}-{max_frames}")

    pickle_files = glob.glob(os.path.join(directory_path, "*.pkl"))
    if not pickle_files:
        print("No pickle files found")
        return

    print(f"Found {len(pickle_files)} pickle files")
    if keep_mode == "move":
        if output_dir is None:
            output_dir = os.path.join(directory_path, "filtered_out")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Out-of-range files will be moved to: {output_dir}")
    else:
        print("Out-of-range files will be deleted")

    total_files = len(pickle_files)
    kept_files = 0
    removed_files = 0
    error_files = 0
    frame_counts = []

    for file_path in tqdm(pickle_files, desc="Processing files"):
        try:
            data = joblib.load(file_path)
            filename = os.path.basename(file_path)
            num_frames = get_frame_count(data)
            frame_counts.append(num_frames)

            if min_frames <= num_frames <= max_frames:
                kept_files += 1
                print(f"Keep: {filename} ({num_frames} frames)")
            else:
                if keep_mode == "move":
                    destination = os.path.join(output_dir, filename)
                    shutil.move(file_path, destination)
                    print(f"Move: {filename} ({num_frames} frames) -> {output_dir}")
                else:
                    os.remove(file_path)
                    print(f"Delete: {filename} ({num_frames} frames)")
                removed_files += 1
        except Exception as error:
            print(f"Failed to process {file_path}: {error}")
            error_files += 1

    print("\n" + "=" * 50)
    print("Filter summary")
    print("=" * 50)
    print(f"Total: {total_files}")
    print(f"Kept: {kept_files}")
    print(f"Removed: {removed_files}")
    print(f"Errors: {error_files}")

    frame_counts = np.asarray(frame_counts)
    if frame_counts.size:
        print("\nFrame statistics:")
        print(f"  Minimum: {frame_counts.min()}")
        print(f"  Maximum: {frame_counts.max()}")
        print(f"  Mean: {frame_counts.mean():.2f}")
        print(f"  Median: {np.median(frame_counts):.1f}")

    report_path = os.path.join(directory_path, "filter_report.txt")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"Frame range: {min_frames}-{max_frames}\n")
        report.write(f"Total: {total_files}\n")
        report.write(f"Kept: {kept_files}\n")
        report.write(f"Removed: {removed_files}\n")
        report.write(f"Errors: {error_files}\n")
        if frame_counts.size:
            report.write("\nFrame statistics:\n")
            report.write(f"  Minimum: {frame_counts.min()}\n")
            report.write(f"  Maximum: {frame_counts.max()}\n")
            report.write(f"  Mean: {frame_counts.mean():.2f}\n")
            report.write(f"  Median: {np.median(frame_counts):.1f}\n")

    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter motion files by frame count"
    )
    parser.add_argument("--dir", help="Directory containing pickle files")
    parser.add_argument("--min", type=int, default=120, help="Minimum frames")
    parser.add_argument("--max", type=int, default=150, help="Maximum frames")
    parser.add_argument(
        "--mode",
        choices=["move", "delete"],
        default="move",
        help="Action for out-of-range files",
    )
    parser.add_argument("--output", help="Destination when mode is move")
    args = parser.parse_args()

    directory_path = args.dir
    if not directory_path:
        directory_path = "humanoidverse/data/motions/adam/Arm Movements"
        print(f"Using default directory: {directory_path}")

    filter_motion_data(
        directory_path=directory_path,
        min_frames=args.min,
        max_frames=args.max,
        keep_mode=args.mode,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
