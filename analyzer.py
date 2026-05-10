"""
analyzer.py — Pose analysis engine.

Responsibilities:
  - Define ANGLE_INDICES (shared truth for all landmark triplets).
  - Compute joint angles from MediaPipe landmarks (3-D world preferred).
  - Compare measured angles to dataset-derived references.
  - Return a structured FeedbackResult (score, per-joint details, worst joint).

This module has no OpenCV dependency and no side effects at import.
It can be unit-tested independently of the webcam loop.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
#  ANGLE INDICES
#  angle_name → [p1, p2 (vertex), p3]  MediaPipe landmark indices.
#  These are physical constants of the skeleton model — they belong
#  in code, not in config.json.
# ─────────────────────────────────────────────────────────────────
ANGLE_INDICES: dict[str, list[int]] = {
    "Left Knee":      [23, 25, 27],   # hip-knee-ankle
    "Right Knee":     [24, 26, 28],
    "Left Hip":       [11, 23, 25],   # shoulder-hip-knee
    "Right Hip":      [12, 24, 26],
    "Left Elbow":     [11, 13, 15],   # shoulder-elbow-wrist
    "Right Elbow":    [12, 14, 16],
    "Left Shoulder":  [ 7, 11, 13],   # ear-shoulder-elbow
    "Right Shoulder": [ 8, 12, 14],
}


# ─────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────
@dataclass
class JointFeedback:
    """Feedback for a single joint angle."""
    name:     str
    measured: float          # degrees, from current frame
    target:   float          # degrees, from dataset reference
    diff:     float          # abs(measured - target)
    tol:      float          # tolerance from dataset (IQR)
    color:    tuple          # BGR color for display


@dataclass
class FeedbackResult:
    """Complete feedback for one frame."""
    pose_name:   str
    score:       int                          # 0-100
    joints:      list[JointFeedback] = field(default_factory=list)
    worst_joint: Optional[JointFeedback] = None
    n_visible:   int = 0                      # how many joints were detected


# ─────────────────────────────────────────────────────────────────
#  MATHS
# ─────────────────────────────────────────────────────────────────
def angle_between(a, b, c) -> Optional[float]:
    """
    Angle in degrees at vertex b, between vectors b→a and b→c.
    Accepts 2-D [x,y] or 3-D [x,y,z] coordinates.
    """
    a, b, c = (np.array(x, dtype=float) for x in (a, b, c))
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _score_color(diff: float, tol: float, thresholds: dict) -> tuple:
    """BGR color based on how far diff is from the tolerance."""
    green_t  = thresholds.get("green",  75)
    orange_t = thresholds.get("orange", 50)

    # Map diff/tol ratio to a score: 0 = perfect, 1 = at tolerance, >1 = off
    ratio = diff / max(tol, 1e-6)
    if ratio <= 1.0:
        return (50, 220, 80)    # green
    elif ratio <= 1.8:
        return (30, 180, 255)   # orange
    else:
        return (60, 60, 240)    # red (BGR)


def _joint_score(diff: float, tol: float) -> float:
    """Individual score for one joint: 100 at diff=0, 70 at diff=tol, 0 at diff≈3.33*tol."""
    return float(np.clip(100.0 - (diff / max(tol, 1e-6)) * 30.0, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────
def compute_feedback(
    lm_2d,
    lm_world,
    pose_name: str,
    refs: dict,
    config: dict,
) -> FeedbackResult:
    """
    Compute joint angles and feedback for the current frame.

    Parameters
    ----------
    lm_2d     : list of NormalizedLandmark — from results.pose_landmarks[0].
                Has .x (0-1), .y (0-1), .visibility.
    lm_world  : list of Landmark — from results.pose_world_landmarks[0].
                Has .x, .y, .z in metres (hip-centred). No .visibility.
                Pass None to fall back to 2-D computation.
    pose_name : canonical pose name (key in config["poses"]).
    refs      : loaded pose_references.json as a dict.
    config    : loaded config.json as a dict.

    Returns
    -------
    FeedbackResult with score, per-joint details, and worst joint.
    """
    vis_min    = config["feedback"].get("landmark_visibility_min", 0.5)
    thresholds = config["feedback"].get("score_thresholds", {"green": 75, "orange": 50})
    angle_keys = config["poses"][pose_name]["angle_keys"]
    pose_refs  = refs.get(pose_name, {})

    use_3d = lm_world is not None and len(lm_world) >= 33

    joints     = []
    ind_scores = []

    for angle_name in angle_keys:
        if angle_name not in ANGLE_INDICES:
            continue
        ref = pose_refs.get(angle_name)
        if ref is None:
            continue   # no reference data for this joint — skip silently

        p1, p2, p3 = ANGLE_INDICES[angle_name]

        # Visibility check (2-D landmarks carry visibility)
        try:
            vis = [getattr(lm_2d[i], "visibility", 1.0) for i in (p1, p2, p3)]
            if any(v < vis_min for v in vis):
                continue
        except (IndexError, AttributeError):
            continue

        # Angle computation — 3-D preferred
        try:
            if use_3d:
                pts = [[lm_world[i].x, lm_world[i].y, lm_world[i].z]
                       for i in (p1, p2, p3)]
            else:
                pts = [[lm_2d[i].x, lm_2d[i].y] for i in (p1, p2, p3)]

            measured = angle_between(*pts)
            if measured is None:
                continue
        except (IndexError, AttributeError):
            continue

        target = float(ref["target"])
        tol    = float(ref["tolerance"])
        diff   = abs(measured - target)
        color  = _score_color(diff, tol, thresholds)
        s      = _joint_score(diff, tol)

        joints.append(JointFeedback(
            name=angle_name, measured=measured,
            target=target, diff=diff,
            tol=tol, color=color,
        ))
        ind_scores.append(s)

    score      = int(np.mean(ind_scores)) if ind_scores else 0
    worst      = max(joints, key=lambda j: j.diff) if joints else None

    return FeedbackResult(
        pose_name=pose_name,
        score=score,
        joints=joints,
        worst_joint=worst,
        n_visible=len(joints),
    )