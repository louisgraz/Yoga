"""
session.py — Session management.

Tracks:
  - Hold timer: user must maintain score >= min_hold_score for hold_duration_s
    seconds to count a successful rep.
  - Rep counter: incremented on every successful hold.
  - Session log: every successful rep is appended to a CSV file in log_dir.

Usage
─────
  session = Session(config)

  # Called every frame:
  progress, just_completed = session.update(score, pose_name)
  # progress: 0.0–1.0 (arc fill in UI)
  # just_completed: True for exactly one frame when a hold finishes
"""

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────
@dataclass
class RepRecord:
    timestamp:  str
    pose_name:  str
    score:      int
    hold_s:     float
    angles:     dict = field(default_factory=dict)   # angle_name → measured°


# ─────────────────────────────────────────────────────────────────
class Session:
    """
    Encapsulates all per-session state.

    The hold timer starts as soon as score >= min_hold_score and
    resets if the score drops below that threshold before completing.
    """

    def __init__(self, config: dict):
        fb  = config["feedback"]
        ses = config["session"]

        self._hold_duration: float = float(fb.get("hold_duration_s", 5.0))
        self._min_score:     int   = int(fb.get("min_hold_score",    70))
        self._log_dir:       str   = ses.get("log_dir", "sessions")
        self._log_enabled:   bool  = ses.get("enabled", True)

        self._hold_start:  Optional[float] = None   # time.time() when hold began
        self._hold_pose:   Optional[str]   = None   # pose being held
        self._rep_counts:  dict[str, int]  = {}     # pose_name → n_reps
        self._reps:        list[RepRecord] = []
        self._log_path:    Optional[str]   = None

        if self._log_enabled:
            self._init_log()

    # ── Public API ──────────────────────────────────────────────

    def update(
        self,
        score: int,
        pose_name: str,
        angles: Optional[dict] = None,
    ) -> tuple[float, bool]:
        """
        Call once per frame.

        Returns
        -------
        (progress, just_completed)
          progress       : 0.0–1.0 — how full the hold arc should be drawn.
          just_completed : True for exactly one frame when the hold finishes.
        """
        holding_correct_pose = (score >= self._min_score)

        # If pose changed while holding, reset
        if self._hold_pose and self._hold_pose != pose_name:
            self._hold_start = None
            self._hold_pose  = None

        if not holding_correct_pose:
            self._hold_start = None
            self._hold_pose  = None
            return 0.0, False

        # Start timer
        if self._hold_start is None:
            self._hold_start = time.time()
            self._hold_pose  = pose_name

        elapsed  = time.time() - self._hold_start
        progress = min(1.0, elapsed / self._hold_duration)

        if progress >= 1.0:
            # Rep completed — reset and log
            self._hold_start = None
            self._hold_pose  = None
            self._rep_counts[pose_name] = self._rep_counts.get(pose_name, 0) + 1

            record = RepRecord(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pose_name=pose_name,
                score=score,
                hold_s=round(self._hold_duration, 1),
                angles=angles or {},
            )
            self._reps.append(record)
            if self._log_enabled and self._log_path:
                self._write_rep(record)

            return 1.0, True

        return progress, False

    def rep_count(self, pose_name: Optional[str] = None) -> int:
        """Total reps, or reps for a specific pose."""
        if pose_name:
            return self._rep_counts.get(pose_name, 0)
        return sum(self._rep_counts.values())

    def summary(self) -> dict:
        """Return a summary dict for display or logging."""
        return {
            "total_reps":   self.rep_count(),
            "by_pose":      dict(self._rep_counts),
            "session_file": self._log_path,
        }

    # ── Internal ─────────────────────────────────────────────────

    def _init_log(self):
        os.makedirs(self._log_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(self._log_dir, f"session_{ts}.csv")

        with open(self._log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "pose_name", "score", "hold_s",
                *[f"{k} (deg)" for k in [
                    "Left Knee", "Right Knee",
                    "Left Hip",  "Right Hip",
                    "Left Elbow", "Right Elbow",
                    "Left Shoulder", "Right Shoulder",
                ]],
            ])

    def _write_rep(self, record: RepRecord):
        ANGLE_COLS = [
            "Left Knee", "Right Knee",
            "Left Hip",  "Right Hip",
            "Left Elbow", "Right Elbow",
            "Left Shoulder", "Right Shoulder",
        ]
        row = [
            record.timestamp, record.pose_name,
            record.score, record.hold_s,
            *[round(record.angles.get(k, float("nan")), 1) for k in ANGLE_COLS],
        ]
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)