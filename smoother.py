"""
smoother.py — Temporal smoothing for pose feedback signals.
============================================================
Solves the core UX problem: at 30 fps, small body movements cause rapid
flickering in joint colours and numbers, which is distracting and prevents
the user from reading the feedback while holding a pose.

Three mechanisms, all configurable in config.json → "smoothing":

  1. Angle smoothing   — rolling mean over `angle_window_frames` frames.
                         Smooths out sensor noise and micro-tremors.
                         Default: 15 frames ≈ 0.5 s at 30 fps.

  2. Colour hysteresis — a joint colour only changes after the joint has
                         stayed in the new zone for `color_debounce_frames`
                         consecutive frames.  Prevents flickering at zone
                         boundaries (e.g. joint oscillating around the
                         green/orange threshold).
                         Default: 20 frames ≈ 0.67 s.

  3. Panel throttling  — the text panel (numbers, deviation values) refreshes
                         at most once every `panel_update_interval_s` seconds.
                         The skeleton colour overlay still updates every frame
                         (using the already-stabilised hysteresis colours).
                         Default: 0.5 s.

Usage
-----
    smoother = FeedbackSmoother(config)

    # called once per frame, AFTER compute_feedback():
    stable_fb, panel_ready = smoother.update(raw_fb)

    # stable_fb  : FeedbackResult with smoothed angles + stable colours.
    # panel_ready: True when enough time has passed to refresh the text panel.

    draw_skeleton(frame, lm_2d, stable_fb.joint_colors())
    if panel_ready:
        draw_right_panel(frame, stable_fb, ...)
    else:
        draw_right_panel(frame, smoother.last_panel_fb, ...)
"""

import time
from collections import deque
from copy import deepcopy

import numpy as np

from analyzer import JointFeedback, FeedbackResult


# ─────────────────────────────────────────────────────────────────
#  COLOUR HELPERS  (duplicated from analyzer to keep smoother
#  independent — avoids a circular import)
# ─────────────────────────────────────────────────────────────────
def _score_color(diff: float, tol: float, thresholds: dict) -> tuple:
    ratio = diff / max(tol, 1e-6)
    if ratio <= 1.0:
        return (50, 220, 80)    # green
    elif ratio <= 1.8:
        return (30, 180, 255)   # orange
    else:
        return (60, 60, 240)    # red


def _joint_score(diff: float, tol: float) -> float:
    return float(np.clip(100.0 - (diff / max(tol, 1e-6)) * 50.0, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────
#  PER-JOINT STATE
# ─────────────────────────────────────────────────────────────────
class _JointState:
    """Tracks rolling angle buffer and colour hysteresis for one joint."""

    def __init__(self, window: int, debounce: int):
        self._buf        = deque(maxlen=window)
        self._debounce   = debounce
        # Current displayed colour (starts undefined until first frame)
        self.stable_color: tuple | None = None
        # Candidate colour and how many consecutive frames it has been seen
        self._candidate:   tuple | None = None
        self._cand_count:  int          = 0

    def push(self, measured: float, target: float,
             tol: float, thresholds: dict) -> tuple[float, float, tuple]:
        """
        Add a new measurement.
        Returns (smooth_measured, smooth_diff, stable_colour).
        """
        self._buf.append(measured)
        smooth_measured = float(np.mean(self._buf))
        smooth_diff     = abs(smooth_measured - target)
        new_color       = _score_color(smooth_diff, tol, thresholds)

        # Hysteresis: only commit to new_color after it appears for
        # `debounce` consecutive frames.
        if new_color == self._candidate:
            self._cand_count += 1
        else:
            self._candidate  = new_color
            self._cand_count = 1

        if self.stable_color is None or self._cand_count >= self._debounce:
            self.stable_color = self._candidate

        return smooth_measured, smooth_diff, self.stable_color


# ─────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────
class FeedbackSmoother:
    """
    Wraps a raw FeedbackResult stream and produces stable, low-noise output.

    Parameters (all from config["smoothing"]):
        angle_window_frames      int   rolling mean window  (default 15)
        color_debounce_frames    int   hysteresis threshold  (default 20)
        panel_update_interval_s  float text-panel refresh rate (default 0.5)
    """

    def __init__(self, config: dict):
        sm = config.get("smoothing", {})
        fb = config.get("feedback", {})

        self._window    : int   = int(sm.get("angle_window_frames",     15))
        self._debounce  : int   = int(sm.get("color_debounce_frames",   20))
        self._interval  : float = float(sm.get("panel_update_interval_s", 0.5))
        self._thresholds: dict  = fb.get("score_thresholds",
                                         {"green": 75, "orange": 50})

        self._joint_states: dict[str, _JointState] = {}

        # Panel throttle state
        self._last_panel_time: float             = 0.0
        self.last_panel_fb:    FeedbackResult | None = None

    # ── Public API ───────────────────────────────────────────────

    def update(
        self,
        raw_fb: FeedbackResult,
    ) -> tuple[FeedbackResult, bool]:
        """
        Process one frame.

        Parameters
        ----------
        raw_fb : FeedbackResult from analyzer.compute_feedback()

        Returns
        -------
        stable_fb   : FeedbackResult with smoothed angles + stable colours.
                      Use this for skeleton drawing (every frame).
        panel_ready : True when the text panel should be refreshed.
                      Use stable_fb for the panel when True;
                      use smoother.last_panel_fb otherwise.
        """
        stable_joints: list[JointFeedback] = []

        for j in raw_fb.joints:
            state = self._joint_states.setdefault(
                j.name, _JointState(self._window, self._debounce)
            )
            smooth_m, smooth_d, color = state.push(
                j.measured, j.target, j.tol, self._thresholds
            )
            stable_joints.append(JointFeedback(
                name     = j.name,
                measured = smooth_m,
                target   = j.target,
                diff     = smooth_d,
                tol      = j.tol,
                color    = color,
            ))

        # Recompute score from stabilised diffs
        scores = [_joint_score(j.diff, j.tol) for j in stable_joints]
        stable_score = int(np.mean(scores)) if scores else 0
        worst = max(stable_joints, key=lambda j: j.diff) if stable_joints else None

        stable_fb = FeedbackResult(
            pose_name             = raw_fb.pose_name,
            score                 = stable_score,
            joints                = stable_joints,
            worst_joint           = worst,
            n_visible             = raw_fb.n_visible,
            insufficient_coverage = raw_fb.insufficient_coverage,
        )

        # Panel throttle
        now         = time.time()
        panel_ready = (now - self._last_panel_time) >= self._interval

        if panel_ready:
            self._last_panel_time = now
            self.last_panel_fb    = stable_fb

        # Ensure last_panel_fb is always populated (first frame)
        if self.last_panel_fb is None:
            self.last_panel_fb = stable_fb

        return stable_fb, panel_ready

    def reset(self):
        """Call when the user switches pose — clears all buffers."""
        self._joint_states.clear()
        self._last_panel_time = 0.0
        self.last_panel_fb    = None