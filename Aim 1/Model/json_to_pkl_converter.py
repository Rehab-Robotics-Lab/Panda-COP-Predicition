#!/usr/bin/env python3
"""
Standalone script to convert mmpose JSON annotations to PKL/CSV files.

This script processes JSON files from pose estimation and converts them to
pandas DataFrame format (saved as both PKL and CSV).

Usage:
    python json_to_pkl_converter.py --input-dir /path/to/json/files --output-dir /path/to/output

    Or for a single file:
    python json_to_pkl_converter.py --input-file video.json --output-file video.pkl --video-path video.mp4
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from moviepy import VideoFileClip
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)

# ============================================================================
# HELPER FUNCTIONS (from processing.py)
# ============================================================================


def get_best_instance(instances):
    """
    Given a list of instances, return the index of the instance with highest confidence keypoints.
    """
    best_score = 0
    n_instance = 0

    for e, instance in enumerate(instances):
        confidence = instance.get("keypoint_scores", [])

        if len(confidence) == 17:
            score = sum(confidence)

            if score > best_score:
                best_score = score
                n_instance = e

    return n_instance


def reorder_keypoints(keypoints, confidence_scores):
    """
    Reorder the keypoints to the OpenPose format.
    """
    keypoints = [keypoints[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
    confidence_scores = [confidence_scores[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
    return keypoints, confidence_scores


def rescale_keypoints(keypoints, scale):
    """
    Rescale the keypoints by the given scale.
    """
    keypoints = [(x * scale, y * scale) for (x, y) in keypoints]
    return keypoints


def convert_coco_to_openpose(coco_keypoints, confidence_scores):
    """
    Convert COCO keypoints to OpenPose keypoints with the neck keypoint as
    the midpoint between the two shoulders.
    """
    (
        nose,
        left_eye,
        right_eye,
        left_ear,
        right_ear,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
    ) = coco_keypoints

    # Calculate the neck as the midpoint between shoulders
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
    neck = (neck_x, neck_y)

    (
        c_nose,
        c_left_eye,
        c_right_eye,
        c_left_ear,
        c_right_ear,
        c_left_shoulder,
        c_right_shoulder,
        c_left_elbow,
        c_right_elbow,
        c_left_wrist,
        c_right_wrist,
        c_left_hip,
        c_right_hip,
        c_left_knee,
        c_right_knee,
        c_left_ankle,
        c_right_ankle,
    ) = confidence_scores

    # Calculate neck confidence
    c_neck = (c_left_shoulder + c_right_shoulder) / 2

    # Construct the OpenPose keypoints including the neck
    openpose_keypoints = [
        nose,
        left_eye,
        right_eye,
        left_ear,
        right_ear,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        neck,
    ]

    openpose_confidences = [
        c_nose,
        c_left_eye,
        c_right_eye,
        c_left_ear,
        c_right_ear,
        c_left_shoulder,
        c_right_shoulder,
        c_left_elbow,
        c_right_elbow,
        c_left_wrist,
        c_right_wrist,
        c_left_hip,
        c_right_hip,
        c_left_knee,
        c_right_knee,
        c_left_ankle,
        c_right_ankle,
        c_neck,
    ]

    openpose_keypoints, confidences = reorder_keypoints(openpose_keypoints, openpose_confidences)
    openpose_keypoints = rescale_keypoints(openpose_keypoints, 1)

    return openpose_keypoints, confidences


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

# OpenPose keypoint mapping
KP_MAPPING = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
}

COLUMNS = ["video_number", "video", "bp", "frame", "x", "y", "c", "fps", "pixel_x", "pixel_y", "time", "part_idx"]


def get_video_metadata(video_path):
    """Extract video metadata (fps, width, height)"""
    try:
        clip = VideoFileClip(str(video_path))
        width, height = clip.size
        fps = clip.fps
        clip.close()
        return fps, width, height
    except Exception as e:
        logging.error(f"Failed to extract video metadata: {e}")
        return None, None, None


def convert_json_to_pkl(
    json_file,
    output_pkl=None,
    output_csv=None,
    video_path=None,
    fps=None,
    width=None,
    height=None,
    use_best_instance=True,
):
    """
    Convert a single JSON annotation file to PKL and CSV.

    Args:
        json_file: Path to JSON file
        output_pkl: Path for output PKL file (optional)
        output_csv: Path for output CSV file (optional)
        video_path: Path to original video (for metadata extraction)
        fps: Video FPS (if known, otherwise will extract from video)
        width: Video width (if known)
        height: Video height (if known)
        use_best_instance: If True, use best instance; if False, use first

    Returns:
        DataFrame if successful, None otherwise
    """
    json_file = Path(json_file)

    # Load JSON
    try:
        with open(json_file, "r") as f:
            frames = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON {json_file}: {e}")
        return None

    if not frames:
        logging.error(f"No frames in JSON: {json_file}")
        return None

    # Get video metadata if not provided
    if fps is None or width is None or height is None:
        if video_path is not None:
            fps_vid, width_vid, height_vid = get_video_metadata(video_path)
            fps = fps if fps is not None else fps_vid
            width = width if width is not None else width_vid
            height = height if height is not None else height_vid
        else:
            logging.warning("Video metadata not provided and no video path given")
            fps = fps or 30.0
            width = width or 1920
            height = height or 1080

    # Generate video identifiers
    video_name = json_file.stem
    video_number = hash(str(json_file)) % 100000

    # Process frames
    interim = []
    frames_processed = 0
    frames_with_data = 0

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        frames_processed += 1
        frame_id = frame.get("frame_id", frames_processed - 1)

        if "instances" not in frame or len(frame["instances"]) == 0:
            continue

        frames_with_data += 1

        # Select instance
        try:
            if use_best_instance:
                instance_id = get_best_instance(frame["instances"])
            else:
                instance_id = 0
        except Exception as e:
            logging.warning(f"Error selecting instance in frame {frame_id}: {e}")
            instance_id = 0

        if instance_id >= len(frame["instances"]):
            continue

        inst = frame["instances"][instance_id]

        # Extract keypoints
        if "keypoints" not in inst or "keypoint_scores" not in inst:
            continue

        keypoints = inst["keypoints"]
        confidence = inst["keypoint_scores"]

        # Convert to OpenPose format
        try:
            keypoints, confidence = convert_coco_to_openpose(keypoints, confidence)
        except Exception as e:
            logging.warning(f"Error converting keypoints in frame {frame_id}: {e}")
            continue

        # Add each keypoint to table
        for part_idx, (x, y) in enumerate(keypoints):
            if part_idx not in KP_MAPPING:
                continue

            bp = KP_MAPPING[part_idx]
            c = confidence[part_idx] if part_idx < len(confidence) else 0.0
            t = frame_id / fps

            row = [video_number, video_name, bp, frame_id, x, y, c, fps, width, height, t, part_idx]
            interim.append(row)

    logging.info(
        f"Processed {frames_processed} frames, {frames_with_data} with detections, " f"{len(interim)} total keypoints"
    )

    if not interim:
        logging.error("No keypoint data extracted from JSON")
        return None

    # Create DataFrame
    df = pd.DataFrame(interim, columns=COLUMNS)

    # Save files
    if output_pkl:
        output_pkl = Path(output_pkl)
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_pkl)
        logging.info(f"Saved PKL: {output_pkl}")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logging.info(f"Saved CSV: {output_csv}")

    return df


def process_directory(input_dir, output_dir, video_dir=None, recursive=True, use_best_instance=True):
    """
    Process all JSON files in a directory.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory for output PKL/CSV files
        video_dir: Directory containing original videos (optional)
        recursive: If True, search subdirectories
        use_best_instance: If True, use best instance; if False, use first
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all JSON files
    if recursive:
        json_files = list(input_dir.rglob("*.json"))
    else:
        json_files = list(input_dir.glob("*.json"))

    logging.info(f"Found {len(json_files)} JSON files to process")

    success_count = 0
    failed_count = 0

    for i, json_file in enumerate(json_files, 1):
        logging.info(f"Processing {i}/{len(json_files)}: {json_file.name}")

        # Determine output paths (maintain directory structure)
        relative_path = json_file.relative_to(input_dir)
        output_pkl = (output_dir / relative_path).with_suffix(".pkl")
        output_csv = (output_dir / relative_path).with_suffix(".csv")

        # Try to find corresponding video
        video_path = None
        if video_dir:
            video_dir = Path(video_dir)
            video_name = json_file.stem
            for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]:
                potential_video = video_dir / relative_path.parent / f"{video_name}{ext}"
                if potential_video.exists():
                    video_path = potential_video
                    break

        # Convert
        try:
            df = convert_json_to_pkl(
                json_file,
                output_pkl=output_pkl,
                output_csv=output_csv,
                video_path=video_path,
                use_best_instance=use_best_instance,
            )

            if df is not None:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logging.error(f"Failed to process {json_file}: {e}")
            logging.debug(traceback.format_exc())
            failed_count += 1

    logging.info(f"\nProcessing complete:")
    logging.info(f"  Success: {success_count}")
    logging.info(f"  Failed: {failed_count}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Convert mmpose JSON annotations to PKL/CSV format")

    # Single file mode
    parser.add_argument("--input-file", type=str, help="Single JSON file to convert")
    parser.add_argument("--output-file", type=str, help="Output PKL file path (for single file mode)")
    parser.add_argument("--video-path", type=str, help="Path to original video (for metadata extraction)")

    # Directory mode
    parser.add_argument("--input-dir", type=str, help="Directory containing JSON files")
    parser.add_argument("--output-dir", type=str, help="Output directory for PKL/CSV files")
    parser.add_argument("--video-dir", type=str, help="Directory containing original videos (optional)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories")

    # Options
    parser.add_argument("--fps", type=float, help="Video FPS (if not extracting from video)")
    parser.add_argument("--width", type=int, help="Video width in pixels")
    parser.add_argument("--height", type=int, help="Video height in pixels")
    parser.add_argument("--first-instance", action="store_true", help="Use first instance instead of best instance")

    args = parser.parse_args()

    # Single file mode
    if args.input_file:
        if not args.output_file:
            output_pkl = Path(args.input_file).with_suffix(".pkl")
            output_csv = Path(args.input_file).with_suffix(".csv")
        else:
            output_pkl = args.output_file
            output_csv = Path(output_pkl).with_suffix(".csv")

        df = convert_json_to_pkl(
            args.input_file,
            output_pkl=output_pkl,
            output_csv=output_csv,
            video_path=args.video_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_best_instance=not args.first_instance,
        )

        if df is not None:
            logging.info("Conversion successful!")
            return 0
        else:
            logging.error("Conversion failed!")
            return 1

    # Directory mode
    elif args.input_dir and args.output_dir:
        process_directory(
            args.input_dir,
            args.output_dir,
            video_dir=args.video_dir,
            recursive=not args.no_recursive,
            use_best_instance=not args.first_instance,
        )
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
