"""
audio.py — Non-blocking audio tone feedback.
=============================================
Uses winsound (built-in on Windows, zero installation).
Falls back silently on other platforms (no crash, no beep).

Two signals only — kept minimal to avoid information overload
(Schmidt & Lee, 2011: simple signals are better than complex cues):

  beep_success()  — short high tone (880 Hz)
                    Meaning: rep validated, posture was held correctly.

  beep_alert()    — short low tone (330 Hz)
                    Meaning: score has been too low for several seconds.

Both are non-blocking: they run in a background thread so the webcam
loop is never paused.

Frequencies and durations are configurable in config.json → "audio".
"""

import platform
import threading
import time


def _play(freq: int, duration_ms: int):
    """Blocking beep — run this in a daemon thread."""
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(freq, duration_ms)
    else:
        # Fallback for macOS / Linux: terminal bell (no external lib)
        print("\a", end="", flush=True)


def beep_async(freq: int, duration_ms: int):
    """Fire-and-forget beep. Never blocks the caller."""
    threading.Thread(target=_play, args=(freq, duration_ms), daemon=True).start()


# ─────────────────────────────────────────────────────────────────
#  AudioFeedback — stateful wrapper with cooldown management
# ─────────────────────────────────────────────────────────────────

class AudioFeedback:
    """
    Manages when to trigger audio cues during the session.

    Parameters come from config["audio"]:
        enabled                 bool   master switch (default True)
        rep_success_freq        int    Hz for rep-completion tone (default 880)
        rep_success_duration_ms int    ms for rep-completion tone (default 150)
        low_score_alert_freq    int    Hz for alert tone (default 330)
        low_score_alert_duration_ms int ms for alert tone (default 200)
        low_score_alert_delay_s float  seconds below threshold before alert fires (default 3.0)
        low_score_alert_cooldown_s float min seconds between two alerts (default 8.0)
    """

    def __init__(self, config: dict):
        cfg = config.get("audio", {})
        self.enabled            = cfg.get("enabled",                    True)
        self._ok_freq           = int(cfg.get("rep_success_freq",        880))
        self._ok_dur            = int(cfg.get("rep_success_duration_ms", 150))
        self._alert_freq        = int(cfg.get("low_score_alert_freq",    330))
        self._alert_dur         = int(cfg.get("low_score_alert_duration_ms", 200))
        self._alert_delay       = float(cfg.get("low_score_alert_delay_s",     3.0))
        self._alert_cooldown    = float(cfg.get("low_score_alert_cooldown_s",  8.0))
        self._min_hold_score    = int(config["feedback"].get("min_hold_score", 70))

        self._low_since:  float | None = None   # when score first dropped
        self._last_alert: float        = 0.0    # time of last alert beep

    # ── Public API ───────────────────────────────────────────────

    def on_rep_completed(self):
        """Call exactly once when a hold rep is validated."""
        if self.enabled:
            beep_async(self._ok_freq, self._ok_dur)

    def update_score(self, score: int, insufficient_coverage: bool):
        """
        Call every frame with the current smoothed score.
        Fires an alert beep if the score has been below threshold
        for long enough and the cooldown has expired.
        """
        if not self.enabled or insufficient_coverage:
            self._low_since = None
            return

        if score < self._min_hold_score:
            if self._low_since is None:
                self._low_since = time.time()
            elif (time.time() - self._low_since  >= self._alert_delay and
                  time.time() - self._last_alert >= self._alert_cooldown):
                beep_async(self._alert_freq, self._alert_dur)
                self._last_alert = time.time()
        else:
            self._low_since = None

    def reset(self):
        """Call when the user switches pose."""
        self._low_since  = None
        self._last_alert = 0.0