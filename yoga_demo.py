"""
Yoga Pose Feedback System — Demo Script
========================================
Requirements : pip install mediapipe opencv-python numpy
Launch       : python yoga_demo.py

Supported poses (press the corresponding key):
  [1] Warrior I
  [2] Tree Pose
  [3] Triangle
  [4] T-Pose  — calibration (arms out)
  [Q] Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
#  LANDMARKER INIT
# ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)
landmarker = vision.PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
#  REFERENCE ANGLES PER POSE
#  Format: { joint_name: ([p1, p2, p3], target_angle_deg, tolerance_deg) }
#  Angle is measured at p2 between segments p2->p1 and p2->p3.
#
#  NOTE: These values are manually defined from anatomical knowledge.
#  See the README / comments at the bottom for how a real dataset
#  would replace these hardcoded numbers.
# ─────────────────────────────────────────────

POSES = {
    "Warrior I": {
        "Left Knee (front)":    ([23, 25, 27],  90, 15),
        "Right Knee (back)":    ([24, 26, 28], 170, 12),
        "Left Hip":             ([11, 23, 25], 100, 15),
        "Left Elbow":           ([11, 13, 15], 170, 15),
        "Right Elbow":          ([12, 14, 16], 170, 15),
        "Left Shoulder (up)":   ([ 7, 11, 13], 160, 20),
        "Right Shoulder (up)":  ([ 8, 12, 14], 160, 20),
    },
    "Tree Pose": {
        "Right Knee (stand)":   ([24, 26, 28], 175,  8),
        "Right Hip":            ([12, 24, 26], 175, 12),
        "Left Elbow":           ([11, 13, 15],  50, 20),
        "Right Elbow":          ([12, 14, 16],  50, 20),
        "Left Shoulder":        ([ 7, 11, 13],  30, 20),
        "Right Shoulder":       ([ 8, 12, 14],  30, 20),
    },
    "Triangle": {
        "Left Knee":            ([23, 25, 27], 175,  8),
        "Right Knee":           ([24, 26, 28], 175,  8),
        "Left Hip":             ([11, 23, 25],  65, 15),
        "Left Shoulder (up)":   ([ 7, 11, 13],  90, 15),
        "Right Shoulder (dn)":  ([ 8, 12, 14],  90, 15),
        "Left Elbow":           ([11, 13, 15], 175, 10),
        "Right Elbow":          ([12, 14, 16], 175, 10),
    },
    "T-Pose (calib)": {
        "Left Knee":            ([23, 25, 27], 175, 5),
        "Right Knee":           ([24, 26, 28], 175, 5),
        "Left Elbow":           ([11, 13, 15], 175, 5),
        "Right Elbow":          ([12, 14, 16], 175, 5),
        "Left Shoulder":        ([ 7, 11, 13],  90, 5),
        "Right Shoulder":       ([ 8, 12, 14],  90, 5),
        "Left Hip":             ([11, 23, 25], 175, 5),
        "Right Hip":            ([12, 24, 26], 175, 5),
    },
}

POSE_KEYS = list(POSES.keys())

POSE_TIPS = {
    "Warrior I":      "Arms raised, front knee at 90 deg, back leg straight",
    "Tree Pose":      "Balance on one leg, hands in prayer position",
    "Triangle":       "Wide stance, arms aligned, tilt torso sideways",
    "T-Pose (calib)": "Stand upright, arms out — use this to check detection",
}


# ─────────────────────────────────────────────
#  STICK FIGURE REFERENCE POSITIONS
#  Normalized coords (0-100) per landmark index.
#  Used to draw a small pose thumbnail in the right panel.
#
#  Indices: 0=nose, 11=LShoulder, 12=RShoulder,
#           13=LElbow, 14=RElbow, 15=LWrist, 16=RWrist,
#           23=LHip, 24=RHip, 25=LKnee, 26=RKnee,
#           27=LAnkle, 28=RAnkle
# ─────────────────────────────────────────────

POSE_THUMBNAILS = {
    "Warrior I": {
        0:  (50,  6),
        11: (42, 26),  12: (58, 26),
        13: (36, 10),  14: (64, 10),
        15: (32,  0),  16: (68,  0),
        23: (44, 57),  24: (56, 54),
        25: (30, 82),  26: (73, 72),
        27: (24, 108), 28: (83, 97),
    },
    "Tree Pose": {
        0:  (50,  6),
        11: (38, 26),  12: (62, 26),
        13: (44, 42),  14: (56, 42),
        15: (50, 52),  16: (50, 52),
        23: (46, 63),  24: (54, 63),
        25: (34, 80),  26: (54, 88),
        27: (38, 88),  28: (54, 113),
    },
    "Triangle": {
        0:  (44, 10),
        11: (28, 30),  12: (66, 42),
        13: (16, 24),  14: (76, 60),
        15: ( 8, 18),  16: (82, 73),
        23: (36, 63),  24: (60, 63),
        25: (26, 88),  26: (68, 88),
        27: (16, 113), 28: (78, 113),
    },
    "T-Pose (calib)": {
        0:  (50,  6),
        11: (40, 28),  12: (60, 28),
        13: (22, 28),  14: (78, 28),
        15: ( 6, 28),  16: (94, 28),
        23: (46, 63),  24: (54, 63),
        25: (46, 88),  26: (54, 88),
        27: (46, 113), 28: (54, 113),
    },
}

STICK_CONNECTIONS = [
    ( 0, 11), ( 0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle in degrees at point b, between vectors b->a and b->c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def get_landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def score_color(diff, tolerance):
    """Green = within tolerance | orange = close | red = off."""
    if diff <= tolerance:
        return (50, 220, 80)
    elif diff <= tolerance * 1.8:
        return (30, 180, 255)
    else:
        return (60, 60, 240)


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_pose_thumbnail(frame, pose_name, box_x, box_y, box_w=130, box_h=165):
    """
    Draw a small stick-figure of the target pose inside a panel box.
    The user can glance at it to know what position to aim for.
    """
    pts = POSE_THUMBNAILS.get(pose_name)
    if pts is None:
        return

    draw_rounded_rect(frame, box_x, box_y, box_x + box_w, box_y + box_h,
                      8, (30, 30, 30), 0.78)
    cv2.putText(frame, "Target pose",
                (box_x + 8, box_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)

    margin = 10
    draw_w  = box_w - margin * 2
    draw_h  = box_h - margin - 20   # 20 px for the label

    def to_px(nx, ny):
        x = box_x + margin + int(nx / 100 * draw_w)
        y = box_y + 20     + int(ny / 100 * draw_h)
        return (x, y)

    for a_idx, b_idx in STICK_CONNECTIONS:
        if a_idx in pts and b_idx in pts:
            cv2.line(frame, to_px(*pts[a_idx]), to_px(*pts[b_idx]),
                     (130, 155, 230), 1)

    for idx, (nx, ny) in pts.items():
        r = 4 if idx == 0 else 3   # slightly bigger dot for the head
        cv2.circle(frame, to_px(nx, ny), r, (205, 210, 255), -1)


# ─────────────────────────────────────────────
#  MAIN UI DRAW
# ─────────────────────────────────────────────

SKELETON_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (16, 18), (16, 20),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (11, 23), (12, 24),
    ( 0, 11), ( 0, 12),
]


def draw_ui(frame, results, pose_name, pose_def, fps):
    h, w = frame.shape[:2]

    if not results.pose_landmarks:
        cv2.putText(frame, "No person detected — move closer",
                    (w // 2 - 230, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 240), 2)
        return frame, 0, []

    lms = results.pose_landmarks.landmark

    # ── Compute joint angles ──────────────────
    feedbacks  = []        # (name, measured, target, diff, color, tol)
    joint_cols = {}        # landmark_idx -> BGR color

    for name, (indices, target, tol) in pose_def.items():
        try:
            a = get_landmark_xy(lms, indices[0], w, h)
            b = get_landmark_xy(lms, indices[1], w, h)
            c = get_landmark_xy(lms, indices[2], w, h)
            measured = angle_between(a, b, c)
            diff     = abs(measured - target)
            col      = score_color(diff, tol)
            feedbacks.append((name, measured, target, diff, col, tol))
            joint_cols[indices[1]] = col
        except Exception:
            pass

    # ── Global score ──────────────────────────
    score = 0
    if feedbacks:
        ind = [min(100, max(0, 100 - (d / t) * 30)) for _, _, _, d, _, t in feedbacks]
        score = int(np.mean(ind))

    # ── Colored skeleton overlay ──────────────
    for s_idx, e_idx in SKELETON_CONNECTIONS:
        if s_idx >= len(lms) or e_idx >= len(lms):
            continue
        ls, le = lms[s_idx], lms[e_idx]
        if getattr(ls, "visibility", 1) < 0.4 or getattr(le, "visibility", 1) < 0.4:
            continue
        xs, ys = int(ls.x * w), int(ls.y * h)
        xe, ye = int(le.x * w), int(le.y * h)
        cs  = joint_cols.get(s_idx, (200, 200, 200))
        ce  = joint_cols.get(e_idx, (200, 200, 200))
        seg = tuple(int(v) for v in ((np.array(cs) + np.array(ce)) // 2))
        cv2.line(frame, (xs, ys), (xe, ye), seg, 2)

    for idx, lm in enumerate(lms):
        if getattr(lm, "visibility", 1) < 0.4:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        col  = joint_cols.get(idx, (180, 180, 180))
        cv2.circle(frame, (x, y), 5, col, -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

    # ── Right panel ───────────────────────────
    panel_x = w - 310
    draw_rounded_rect(frame, panel_x - 10, 10, w - 5, h - 10, 12, (20, 20, 20), 0.65)

    cv2.putText(frame, pose_name,
                (panel_x, 38), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1)

    score_col = (50, 220, 80) if score >= 75 else \
                (30, 180, 255) if score >= 50 else (60, 60, 240)
    cv2.putText(frame, f"Score: {score}%",
                (panel_x, 65), cv2.FONT_HERSHEY_DUPLEX, 0.70, score_col, 2)

    bw = 282
    cv2.rectangle(frame, (panel_x, 75), (panel_x + bw, 87), (55, 55, 55), -1)
    cv2.rectangle(frame, (panel_x, 75),
                  (panel_x + int(bw * score / 100), 87), score_col, -1)

    # ── Pose thumbnail ────────────────────────
    draw_pose_thumbnail(frame, pose_name, box_x=panel_x, box_y=95)

    # ── Per-joint feedback list ───────────────
    y_off = 275
    for name, measured, target, diff, col, _ in feedbacks:
        if y_off > h - 65:
            break
        cv2.circle(frame, (panel_x + 8, y_off - 4), 5, col, -1)
        cv2.putText(frame, name,
                    (panel_x + 20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        detail = f"{measured:.0f}d  target:{target}d  ({diff:+.0f})"
        cv2.putText(frame, detail,
                    (panel_x + 20, y_off + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
        y_off += 38

    # ── Correction tip ────────────────────────
    if score < 60 and feedbacks:
        worst     = max(feedbacks, key=lambda x: x[3])
        n_w, m_w, t_w, d_w, _, _ = worst
        direction = "increase" if m_w < t_w else "reduce"
        tip       = f"Tip: {direction} {n_w} by {d_w:.0f} deg"
        tip_right = min(w - 320, 660)
        draw_rounded_rect(frame, 10, h - 52, tip_right, h - 10, 8, (20, 20, 20), 0.72)
        cv2.putText(frame, tip, (18, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 200, 255), 1)

    # FPS + hint
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1)
    hint = POSE_TIPS.get(pose_name, "")
    cv2.putText(frame, hint, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    return frame, score, feedbacks


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        print("        Try VideoCapture(1) if you have multiple cameras.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_pose_idx = 0
    prev_time        = time.time()
    frame_ts         = 0

    print("\n═══════════════════════════════════════")
    print("  Yoga Pose Feedback — Demo")
    print("═══════════════════════════════════════")
    print("  Keys:  1-4 = select pose | Q = quit")
    print("  Stand ~1.5 m from the webcam")
    print("═══════════════════════════════════════\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = landmarker.detect_for_video(mp_image, frame_ts)
        frame_ts += 1

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # ── Wrap new-API results into a simple compatible object ──
        class FakeResults:
            pass

        fake = FakeResults()

        if results.pose_landmarks:
            class FakeLandmarks:
                pass

            class LM:
                def __init__(self, x, y, vis):
                    self.x, self.y, self.visibility = x, y, vis

            fake.pose_landmarks          = FakeLandmarks()
            fake.pose_landmarks.landmark = [
                LM(l.x, l.y, l.visibility if hasattr(l, "visibility") else 1.0)
                for l in results.pose_landmarks[0]
            ]
        else:
            fake.pose_landmarks = None

        pose_name = POSE_KEYS[current_pose_idx]
        frame, score, _ = draw_ui(frame, fake, pose_name, POSES[pose_name], fps)

        # Key legend (top-left)
        for i, pn in enumerate(POSE_KEYS):
            col = (50, 220, 80) if i == current_pose_idx else (110, 110, 110)
            cv2.putText(frame, f"[{i+1}] {pn}", (10, 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)

        cv2.imshow("Yoga Pose Feedback — Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            current_pose_idx = 0
        elif key == ord('2'):
            current_pose_idx = 1
        elif key == ord('3'):
            current_pose_idx = 2
        elif key == ord('4'):
            current_pose_idx = 3

    try:
        landmarker.close()
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()