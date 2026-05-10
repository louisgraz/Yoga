"""
build_references.py — Compute pose reference angles from a yoga dataset.
=========================================================================
Output: pose_references.json   (loaded automatically by yoga_demo.py)

USAGE
─────
  Yoga-82 (your case — downloads images from URLs in .txt files):
    python build_references.py --yoga82 Yoga-82/
    python build_references.py --yoga82 Yoga-82/ --max-per-pose 150

  Image folders (any dataset with pose sub-folders containing images):
    python build_references.py --images path/to/dataset/

  CSV with pre-computed MediaPipe keypoints (Kaggle datasets):
    python build_references.py --csv train.csv [test.csv ...]

REQUIREMENTS
────────────
  pip install mediapipe opencv-python numpy requests
  pip install pandas        # only needed for --csv mode

HOW --yoga82 WORKS
──────────────────
  Yoga-82 ships .txt files (one per pose) listing image paths + download URLs.
  This script finds the relevant .txt files, downloads up to --max-per-pose
  images per pose (cached locally so re-runs are instant), runs MediaPipe on
  each, and computes mean ± std angle statistics.

  Expected Yoga-82 directory structure:
    Yoga-82/
      Warrior_I_Pose_or_Virabhadrasana_I_.txt
      Tree_Pose_or_Vriksasana_.txt
      Triangle_Pose_or_Trikonasana_.txt
      yoga_train.txt
      yoga_test.txt
      yoga_dataset_links/   (optional sub-folder — also searched)
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ─────────────────────────────────────────────────────────────────────
#  ANGLE DEFINITIONS  (must stay in sync with yoga_demo.py)
# ─────────────────────────────────────────────────────────────────────
ANGLE_INDICES = {
    "Left Knee":      [23, 25, 27],
    "Right Knee":     [24, 26, 28],
    "Left Hip":       [11, 23, 25],
    "Right Hip":      [12, 24, 26],
    "Left Elbow":     [11, 13, 15],
    "Right Elbow":    [12, 14, 16],
    "Left Shoulder":  [ 7, 11, 13],
    "Right Shoulder": [ 8, 12, 14],
}

MIN_TOLERANCE = 5.0
MIN_SAMPLES   = 8


# ─────────────────────────────────────────────────────────────────────
#  POSE NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────
POSE_NAME_MAP = {
    # Warrior I
    "warrior_i":              "Warrior I",
    "warrior1":               "Warrior I",
    "warrior_one":            "Warrior I",
    "virabhadrasana_i":       "Warrior I",
    "virabhadrasana1":        "Warrior I",
    "warrior_i_pose":         "Warrior I",
    "warrior_i_pose_or_virabhadrasana_i_": "Warrior I",
    # Warrior II  (used as proxy if Warrior I not found)
    "warrior_ii":             "Warrior I",
    "warrior2":               "Warrior I",
    "warrior_ii_pose_or_virabhadrasana_ii_": "Warrior I",
    # Tree
    "tree":                   "Tree Pose",
    "tree_pose":              "Tree Pose",
    "vrksasana":              "Tree Pose",
    "tree_pose_or_vriksasana_": "Tree Pose",
    # Triangle
    "triangle":               "Triangle",
    "triangle_pose":          "Triangle",
    "trikonasana":            "Triangle",
    "extended_triangle":      "Triangle",
    "utthita_trikonasana":    "Triangle",
    "triangle_pose_or_trikonasana_": "Triangle",
    # T-Pose
    "t_pose":                 "T-Pose (calib)",
    "tpose":                  "T-Pose (calib)",
}

def normalise_name(raw: str):
    key = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if key in POSE_NAME_MAP:
        return POSE_NAME_MAP[key]
    for k, v in POSE_NAME_MAP.items():
        if k in key or key in k:
            return v
    return None


# ─────────────────────────────────────────────────────────────────────
#  MATHS
# ─────────────────────────────────────────────────────────────────────
def angle_between(a, b, c):
    a, b, c = (np.array(x, dtype=float) for x in (a, b, c))
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))

def angles_from_lm_list(lm_list):
    out = {}
    for name, (p1, p2, p3) in ANGLE_INDICES.items():
        try:
            a = [lm_list[p1].x, lm_list[p1].y]
            b = [lm_list[p2].x, lm_list[p2].y]
            c = [lm_list[p3].x, lm_list[p3].y]
            ang = angle_between(a, b, c)
            if ang is not None:
                out[name] = ang
        except (IndexError, AttributeError):
            pass
    return out


# ─────────────────────────────────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def init_image_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def run_mediapipe(landmarker, img_bgr):
    """Run pose detection on a BGR image, return angles dict or None."""
    try:
        rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = landmarker.detect(mp_img)
        if res.pose_landmarks:
            return angles_from_lm_list(res.pose_landmarks[0])
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────
#  YOGA-82 MODE
# ─────────────────────────────────────────────────────────────────────
def find_yoga82_txt_files(yoga82_dir: str):
    """
    Find pose .txt files inside yoga82_dir or one level down.
    Excludes yoga_train.txt and yoga_test.txt.
    Returns list of absolute paths.
    """
    patterns = [
        os.path.join(yoga82_dir, "*.txt"),
        os.path.join(yoga82_dir, "*", "*.txt"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    SKIP = {"yoga_train.txt", "yoga_test.txt", "readme.txt", "readme.md"}
    return [
        p for p in candidates
        if os.path.basename(p).lower() not in SKIP
    ]

def process_yoga82(yoga82_dir: str, max_per_pose: int = 100) -> dict:
    try:
        import requests as req_lib
    except ImportError:
        print("[ERROR] 'requests' is required for --yoga82 mode.")
        print("        pip install requests")
        sys.exit(1)

    # ── Find .txt files ──────────────────────────────────────────
    all_txts = find_yoga82_txt_files(yoga82_dir)
    if not all_txts:
        print(f"[ERROR] No .txt files found inside '{yoga82_dir}'.")
        print("        Check that you pointed to the right folder.")
        sys.exit(1)

    print(f"  Found {len(all_txts)} .txt files total")

    # ── Filter to our target poses ───────────────────────────────
    target_files = []
    unmatched    = []
    for txt_path in sorted(all_txts):
        stem      = os.path.splitext(os.path.basename(txt_path))[0]
        pose_name = normalise_name(stem)
        if pose_name:
            target_files.append((txt_path, stem, pose_name))
        else:
            unmatched.append(stem)

    if not target_files:
        print("[ERROR] None of the .txt files matched our target poses.")
        print(f"  First 10 file names: {[os.path.basename(t) for t in all_txts[:10]]}")
        print("  → Add entries to POSE_NAME_MAP at the top of this script.")
        sys.exit(1)

    print(f"  Matched {len(target_files)} target pose(s): "
          f"{sorted(set(pn for _, _, pn in target_files))}")
    if unmatched:
        print(f"  (ignored {len(unmatched)} unrelated poses)")

    # ── Process each pose ────────────────────────────────────────
    landmarker = init_image_landmarker()
    pose_data  = {}

    for txt_path, raw_stem, pose_name in target_files:
        print(f"\n  ── {pose_name}  ({raw_stem})")

        # Parse entries: local_path <TAB> url
        entries = []
        try:
            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        entries.append((parts[0].strip(), parts[1].strip()))
        except Exception as e:
            print(f"    [WARN] Could not read file: {e}")
            continue

        if not entries:
            print("    No valid entries found, skipping.")
            continue

        print(f"    {len(entries)} entries in file — "
              f"attempting up to {max_per_pose} images")

        if pose_name not in pose_data:
            pose_data[pose_name] = {k: [] for k in ANGLE_INDICES}

        collected   = {k: [] for k in ANGLE_INDICES}
        n_detected  = 0
        n_tried     = 0
        n_cached    = 0
        n_failed    = 0
        t_start     = time.time()

        for local_rel, url in entries:
            if n_detected >= max_per_pose:
                break

            n_tried  += 1
            img_bgr   = None

            # 1) Try local cache first
            local_abs = os.path.join(yoga82_dir, local_rel)
            if os.path.isfile(local_abs):
                img_bgr = cv2.imread(local_abs)
                if img_bgr is not None:
                    n_cached += 1

            # 2) Download if not cached
            if img_bgr is None:
                try:
                    resp = req_lib.get(url, timeout=6,
                                       headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        arr    = np.frombuffer(resp.content, np.uint8)
                        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        # Cache for future runs
                        if img_bgr is not None:
                            os.makedirs(os.path.dirname(local_abs), exist_ok=True)
                            cv2.imwrite(local_abs, img_bgr)
                except Exception:
                    n_failed += 1
                    continue

            if img_bgr is None:
                n_failed += 1
                continue

            # 3) Run MediaPipe
            angles = run_mediapipe(landmarker, img_bgr)
            if angles:
                for k, v in angles.items():
                    collected[k].append(v)
                n_detected += 1
            
            # Progress every 10 images
            if n_tried % 10 == 0:
                elapsed = time.time() - t_start
                print(f"    … {n_detected} detected / {n_tried} tried "
                      f"({n_cached} cached, {n_failed} failed)  "
                      f"{elapsed:.0f}s", end="\r")

        elapsed = time.time() - t_start
        print(f"    ✓  {n_detected} poses detected  "
              f"({n_cached} from cache, {n_failed} URL failures)  "
              f"{elapsed:.0f}s          ")

        if n_detected >= MIN_SAMPLES:
            for k, vals in collected.items():
                pose_data[pose_name][k].extend(vals)
        else:
            print(f"    [WARN] Only {n_detected} samples — "
                  f"need at least {MIN_SAMPLES}. Skipping this pose.")

    landmarker.close()
    return pose_data


# ─────────────────────────────────────────────────────────────────────
#  IMAGE FOLDER MODE
# ─────────────────────────────────────────────────────────────────────
def process_image_folder(root_dir: str) -> dict:
    """Walk root_dir sub-folders, run MediaPipe on every image."""
    subfolders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    if not subfolders:
        print("[ERROR] No sub-folders found inside:", root_dir)
        sys.exit(1)

    landmarker = init_image_landmarker()
    pose_data  = {}
    ignored    = []

    for folder in subfolders:
        pose_name = normalise_name(folder)
        if pose_name is None:
            ignored.append(folder)
            continue

        folder_path = os.path.join(root_dir, folder)
        images = [f for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if not images:
            print(f"  skip '{folder}' — no images found")
            continue

        print(f"  '{folder}' → '{pose_name}'  ({len(images)} images) …",
              end="", flush=True)
        collected  = {k: [] for k in ANGLE_INDICES}
        n_detected = 0

        for fname in images:
            try:
                bgr    = cv2.imread(os.path.join(folder_path, fname))
                angles = run_mediapipe(landmarker, bgr) if bgr is not None else None
                if angles:
                    for k, v in angles.items():
                        collected[k].append(v)
                    n_detected += 1
            except Exception:
                pass

        if n_detected < MIN_SAMPLES:
            print(f" ✗  only {n_detected} detections — skipping")
            continue

        if pose_name not in pose_data:
            pose_data[pose_name] = {k: [] for k in ANGLE_INDICES}
        for k, vals in collected.items():
            pose_data[pose_name][k].extend(vals)
        print(f" ✓  {n_detected}/{len(images)} detected")

    landmarker.close()
    if ignored:
        print(f"\n  Ignored folders: {ignored[:8]}")
    return pose_data


# ─────────────────────────────────────────────────────────────────────
#  CSV MODE
# ─────────────────────────────────────────────────────────────────────
def detect_csv_format(df):
    cols      = list(df.columns)
    label_col = next(
        (c for c in ["label","class","pose","pose_name","target","category"] if c in cols),
        cols[0]
    )
    non_label = [c for c in cols if c != label_col]
    n = len(non_label)

    if "x_0" in cols and "y_0" in cols:
        return label_col, lambda row, i: [float(row[f"x_{i}"]), float(row[f"y_{i}"])]
    if "0_x" in cols and "0_y" in cols:
        return label_col, lambda row, i: [float(row[f"{i}_x"]), float(row[f"{i}_y"])]

    LANDMARK_NAMES = [
        "nose","left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_pinky","right_pinky",
        "left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee",
        "left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index",
    ]
    if f"{LANDMARK_NAMES[0]}_x" in cols:
        return label_col, lambda row, i: [
            float(row[f"{LANDMARK_NAMES[i]}_x"]),
            float(row[f"{LANDMARK_NAMES[i]}_y"]),
        ]
    if n == 33 * 4:
        return label_col, lambda row, i: [float(row[non_label[i*4]]), float(row[non_label[i*4+1]])]
    if n == 33 * 2:
        return label_col, lambda row, i: [float(row[non_label[i*2]]), float(row[non_label[i*2+1]])]

    raise ValueError(f"Unrecognised CSV format ({n} non-label cols). First cols: {cols[:12]}")

def process_csv_files(csv_paths: list) -> dict:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas required for --csv mode:  pip install pandas")
        sys.exit(1)

    pose_data = {}
    ignored   = set()

    for path in csv_paths:
        print(f"  Loading {path} …", end="", flush=True)
        df = pd.read_csv(path)
        print(f" {len(df)} rows")

        try:
            label_col, coord_fn = detect_csv_format(df)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        for _, row in df.iterrows():
            pose_name = normalise_name(str(row[label_col]))
            if pose_name is None:
                ignored.add(str(row[label_col]))
                continue

            if pose_name not in pose_data:
                pose_data[pose_name] = {k: [] for k in ANGLE_INDICES}

            for angle_name, (p1, p2, p3) in ANGLE_INDICES.items():
                try:
                    a, b, c = coord_fn(row, p1), coord_fn(row, p2), coord_fn(row, p3)
                    if any(np.isnan(float(v)) for pt in (a, b, c) for v in pt):
                        continue
                    ang = angle_between(a, b, c)
                    if ang is not None:
                        pose_data[pose_name][angle_name].append(ang)
                except Exception:
                    pass

    if ignored:
        print(f"\n  Ignored CSV labels: {sorted(ignored)}")
    return pose_data


# ─────────────────────────────────────────────────────────────────────
#  AGGREGATE  →  JSON
# ─────────────────────────────────────────────────────────────────────
def build_json(pose_data: dict, out_path: str):
    references = {}

    for pose_name, angle_dict in sorted(pose_data.items()):
        entries = {}
        for angle_name, values in angle_dict.items():
            if len(values) < MIN_SAMPLES:
                continue
            mean = float(np.mean(values))
            std  = max(MIN_TOLERANCE, float(np.std(values)))
            entries[angle_name] = {
                "target":    round(mean, 1),
                "tolerance": round(std,  1),
                "n_samples": len(values),
            }
        if entries:
            references[pose_name] = entries
            sample_n = list(entries.values())[0]["n_samples"]
            print(f"  ✓  {pose_name:<24} {len(entries)} angles  "
                  f"(~{sample_n} samples each)")

    if not references:
        print("\n[ERROR] No valid poses extracted.")
        print("  Common causes:")
        print("  • --yoga82: too many dead URLs → try --max-per-pose 200")
        print("  • --images: folder names not matching pose map")
        print("  • --csv: label names not matching pose map")
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(references, f, indent=2)

    print(f"\n✓  Saved  →  {out_path}")
    print(f"   Poses  :  {list(references.keys())}")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build pose_references.json from a yoga dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--yoga82", metavar="DIR",
                      help="Yoga-82 directory containing pose .txt files")
    mode.add_argument("--images", metavar="DIR",
                      help="Root folder with one sub-folder per pose class")
    mode.add_argument("--csv", metavar="FILE", nargs="+",
                      help="CSV file(s) with pre-computed MediaPipe keypoints")

    parser.add_argument("--max-per-pose", type=int, default=100, metavar="N",
                        help="(--yoga82 only) max images to process per pose "
                             "(default: 100)")
    parser.add_argument("--out", default="pose_references.json", metavar="FILE",
                        help="Output JSON path (default: pose_references.json)")

    args = parser.parse_args()

    if args.yoga82:
        if not os.path.isdir(args.yoga82):
            print(f"[ERROR] Directory not found: {args.yoga82}")
            sys.exit(1)
        print(f"\nMode  : Yoga-82\nInput : {args.yoga82}\n"
              f"Max   : {args.max_per_pose} images/pose\n")
        pose_data = process_yoga82(args.yoga82, args.max_per_pose)

    elif args.images:
        if not os.path.isdir(args.images):
            print(f"[ERROR] Directory not found: {args.images}")
            sys.exit(1)
        print(f"\nMode  : images\nInput : {args.images}\n")
        pose_data = process_image_folder(args.images)

    else:
        missing = [p for p in args.csv if not os.path.isfile(p)]
        if missing:
            print(f"[ERROR] File(s) not found: {missing}")
            sys.exit(1)
        print(f"\nMode  : CSV\nInput : {args.csv}\n")
        pose_data = process_csv_files(args.csv)

    print()
    build_json(pose_data, args.out)
    print(f"\nNext:  python yoga_demo.py\n")


if __name__ == "__main__":
    main()