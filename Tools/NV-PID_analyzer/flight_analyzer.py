#!/usr/bin/env python3
"""
Flight Analyzer - Identifies flight phases suitable for PID analysis
Detects autotune sessions and asks user which one to use
Filters out: MANUAL, TAKEOFF, UNKNOWN modes, and low altitude (< 10m)
Analyzes speed scaler effects on PID performance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from log_parser import LogData, LogParser, ModeEvent, AutotuneEvent, AltitudeSample


@dataclass
class AnalysisPhase:
    """A time period suitable for analysis"""

    start_time: float
    end_time: float
    phase_type: str
    mode_name: str
    mode_num: int
    min_altitude_m: float = 0.0
    avg_scaler: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AutotuneSession:
    """Information about an autotune session"""

    index: int
    start_time: float
    end_time: float
    duration_s: float
    has_roll: bool = False
    has_pitch: bool = False
    roll_finished_time: float = 0.0
    pitch_finished_time: float = 0.0


@dataclass
class AutotunePhysics:
    """Physics learned from autotune session"""

    roll_ff_efficiency: float = 0.0
    pitch_ff_efficiency: float = 0.0
    roll_d_limit: float = 0.0
    pitch_d_limit: float = 0.0
    roll_final_p: float = 0.0
    roll_final_i: float = 0.0
    roll_final_d: float = 0.0
    roll_final_ff: float = 0.0
    pitch_final_p: float = 0.0
    pitch_final_i: float = 0.0
    pitch_final_d: float = 0.0
    pitch_final_ff: float = 0.0


class FlightAnalyzer:
    """Analyze flight phases and identify suitable analysis periods"""

    AUTOTUNE_LEVEL_TARGETS = {
        1: (1.00, 20, 20, 1.0),
        2: (0.90, 30, 18, 0.9),
        3: (0.80, 40, 17, 0.8),
        4: (0.70, 50, 16, 0.7),
        5: (0.60, 60, 15, 0.6),
        6: (0.50, 75, 14, 0.5),
        7: (0.30, 90, 12, 0.4),
        8: (0.20, 120, 10, 0.3),
        9: (0.15, 160, 9, 0.25),
        10: (0.10, 210, 8, 0.2),
    }

    MIN_ALTITUDE_M = 10.0
    MODE_SWITCH_BUFFER_S = 2.0  # Skip data this many seconds after mode change

    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.analysis_phases: List[AnalysisPhase] = []
        self.autotune_physics: Optional[AutotunePhysics] = None
        self.selected_session_idx: int = 0

    def get_autotune_sessions(self) -> List[AutotuneSession]:
        """Get list of all autotune sessions"""
        sessions = []
        start_time = None
        has_roll = False
        has_pitch = False
        roll_time = 0.0
        pitch_time = 0.0

        for event in self.log_data.autotune_events:
            if event.event_type == "start":
                start_time = event.time_s
                has_roll = False
                has_pitch = False
            elif event.event_type == "roll_finished":
                has_roll = True
                roll_time = event.time_s
            elif event.event_type == "pitch_finished":
                has_pitch = True
                pitch_time = event.time_s
            elif event.event_type == "stop" and start_time is not None:
                sessions.append(
                    AutotuneSession(
                        index=len(sessions) + 1,
                        start_time=start_time,
                        end_time=event.time_s,
                        duration_s=event.time_s - start_time,
                        has_roll=has_roll,
                        has_pitch=has_pitch,
                        roll_finished_time=roll_time,
                        pitch_finished_time=pitch_time,
                    )
                )
                start_time = None

        return sessions

    def select_autotune_session(self, session_idx: int) -> Tuple[float, float]:
        """
        Select an autotune session and return the analysis time range.
        Returns (start_time, end_time) for post-autotune analysis.
        """
        sessions = self.get_autotune_sessions()

        if not sessions:
            return (0.0, self.log_data.end_time_s)

        if session_idx < 1 or session_idx > len(sessions):
            print(f"Invalid session index. Using session 1.")
            session_idx = 1

        self.selected_session_idx = session_idx - 1
        session = sessions[self.selected_session_idx]

        # Find the next autotune session (if any) to set end time
        if self.selected_session_idx + 1 < len(sessions):
            end_time = sessions[self.selected_session_idx + 1].start_time
        else:
            end_time = self.log_data.end_time_s

        return (session.end_time, end_time)

    def analyze(
        self, session_idx: Optional[int] = None
    ) -> Tuple[List[AnalysisPhase], Optional[AutotunePhysics]]:
        """
        Analyze flight phases.

        If session_idx is provided, use that autotune session.
        Otherwise, automatically select or ask user.
        """
        sessions = self.get_autotune_sessions()

        if sessions and self.log_data.atrp:
            self.autotune_physics = self._extract_autotune_physics()

        mode_timeline = self._build_mode_timeline()
        altitude_lookup = self._build_altitude_lookup()
        scaler_lookup = self._build_scaler_lookup()

        if sessions:
            # Determine which session to use
            if session_idx is None:
                # Use first session by default
                session_idx = 1

            start_time, end_time = self.select_autotune_session(session_idx)
            self.analysis_phases = self._get_post_autotune_phases(
                start_time, end_time, mode_timeline, altitude_lookup, scaler_lookup
            )
        else:
            self.analysis_phases = self._get_all_suitable_phases(
                mode_timeline, altitude_lookup, scaler_lookup
            )

        return self.analysis_phases, self.autotune_physics

    def _build_mode_timeline(self) -> List[Tuple[float, float, int, str]]:
        """Build timeline of (start, end, mode_num, mode_name)"""
        timeline = []

        if not self.log_data.modes:
            return timeline

        sorted_modes = sorted(self.log_data.modes, key=lambda m: m.time_us)

        for i, mode in enumerate(sorted_modes):
            start_s = mode.time_us * 1e-6

            if i + 1 < len(sorted_modes):
                end_s = sorted_modes[i + 1].time_us * 1e-6
            else:
                end_s = self.log_data.end_time_s

            mode_name = LogParser.get_mode_name(mode.mode)
            timeline.append((start_s, end_s, mode.mode, mode_name))

        return timeline

    def _build_altitude_lookup(self) -> Dict[float, float]:
        """Build altitude lookup table {time_s: altitude_m}"""
        return {sample.time_s: sample.altitude_m for sample in self.log_data.altitude}

    def _build_scaler_lookup(self) -> Dict[float, float]:
        """Build speed scaler lookup table {time_s: scaler}"""
        if self.log_data.speed_scaler is None:
            return {}
        return {
            t: s
            for t, s in zip(
                self.log_data.speed_scaler.time, self.log_data.speed_scaler.scaler
            )
        }

    def _get_value_at_time(
        self, time_s: float, lookup: Dict[float, float], default: float
    ) -> float:
        """Get interpolated value at a specific time"""
        if not lookup:
            return default

        times = sorted(lookup.keys())

        if time_s <= times[0]:
            return lookup[times[0]]
        if time_s >= times[-1]:
            return lookup[times[-1]]

        for i, t in enumerate(times):
            if t > time_s:
                t0 = times[i - 1]
                t1 = t
                v0 = lookup[t0]
                v1 = lookup[t1]
                ratio = (time_s - t0) / (t1 - t0)
                return v0 + ratio * (v1 - v0)

        return default

    def _get_altitude_at_time(
        self, time_s: float, altitude_lookup: Dict[float, float]
    ) -> float:
        return self._get_value_at_time(time_s, altitude_lookup, 100.0)

    def _get_scaler_at_time(
        self, time_s: float, scaler_lookup: Dict[float, float]
    ) -> float:
        return self._get_value_at_time(time_s, scaler_lookup, 1.0)

    def _get_min_altitude_in_range(
        self, start_time: float, end_time: float, altitude_lookup: Dict[float, float]
    ) -> float:
        """Get minimum altitude within a time range"""
        if not altitude_lookup:
            return 100.0

        altitudes = [
            alt for t, alt in altitude_lookup.items() if start_time <= t <= end_time
        ]

        if not altitudes:
            # No altitude data in range, use interpolated value at start
            return self._get_altitude_at_time(start_time, altitude_lookup)

        return min(altitudes)

    def _get_post_autotune_phases(
        self,
        start_time: float,
        end_time: float,
        mode_timeline: List[Tuple[float, float, int, str]],
        altitude_lookup: Dict[float, float],
        scaler_lookup: Dict[float, float],
    ) -> List[AnalysisPhase]:
        """Get analysis phases for post-autotune period"""
        phases = []

        for mode_start, mode_end, mode_num, mode_name in mode_timeline:
            # Skip if outside our analysis window
            if mode_end <= start_time or mode_start >= end_time:
                continue

            # Adjust boundaries to analysis window
            phase_start = max(mode_start, start_time)
            phase_end = min(mode_end, end_time)

            # Skip MANUAL, TAKEOFF, and UNKNOWN modes
            if LogParser.should_skip_mode(mode_num):
                continue

            # Skip AUTOTUNE mode itself
            if mode_num == 8:
                continue

            # Check if suitable mode for PID analysis
            if not LogParser.is_suitable_mode(mode_num):
                continue

            # Add buffer after mode change to skip transient data
            phase_start += self.MODE_SWITCH_BUFFER_S
            if phase_start >= phase_end:
                continue

            # Check altitude throughout the phase (not just at start)
            min_altitude = self._get_min_altitude_in_range(
                phase_start, phase_end, altitude_lookup
            )
            if min_altitude < self.MIN_ALTITUDE_M:
                continue

            # Get average scaler for this phase
            avg_scaler = self._get_scaler_at_time(
                (phase_start + phase_end) / 2, scaler_lookup
            )

            phases.append(
                AnalysisPhase(
                    start_time=phase_start,
                    end_time=phase_end,
                    phase_type="post_autotune",
                    mode_name=mode_name,
                    mode_num=mode_num,
                    min_altitude_m=min_altitude,
                    avg_scaler=avg_scaler,
                )
            )

        return phases

    def _get_all_suitable_phases(
        self,
        mode_timeline: List[Tuple[float, float, int, str]],
        altitude_lookup: Dict[float, float],
        scaler_lookup: Dict[float, float],
    ) -> List[AnalysisPhase]:
        """Get all suitable flight phases when no autotune exists"""
        phases = []

        for mode_start, mode_end, mode_num, mode_name in mode_timeline:
            if LogParser.should_skip_mode(mode_num):
                continue

            if mode_num == 8:
                continue

            if LogParser.is_suitable_mode(mode_num):
                # Add buffer after mode change
                phase_start = mode_start + self.MODE_SWITCH_BUFFER_S
                if phase_start >= mode_end:
                    continue

                # Check altitude throughout the phase
                min_altitude = self._get_min_altitude_in_range(
                    phase_start, mode_end, altitude_lookup
                )
                if min_altitude >= self.MIN_ALTITUDE_M:
                    avg_scaler = self._get_scaler_at_time(
                        (phase_start + mode_end) / 2, scaler_lookup
                    )
                    phases.append(
                        AnalysisPhase(
                            start_time=phase_start,
                            end_time=mode_end,
                            phase_type="normal_flight",
                            mode_name=mode_name,
                            mode_num=mode_num,
                            min_altitude_m=min_altitude,
                            avg_scaler=avg_scaler,
                        )
                    )

        return phases

    def _extract_autotune_physics(self) -> AutotunePhysics:
        """Learn aircraft physics from autotune ATRP data"""
        physics = AutotunePhysics()

        if self.log_data.atrp is None or len(self.log_data.atrp.time) == 0:
            return physics

        roll_mask = self.log_data.atrp.axis == 0
        pitch_mask = self.log_data.atrp.axis == 1

        if np.any(roll_mask):
            roll_data = self.log_data.atrp
            roll_ff = roll_data.FF[roll_mask]
            roll_p = roll_data.P[roll_mask]
            roll_i = roll_data.I[roll_mask]
            roll_d = roll_data.D[roll_mask]
            roll_actuator = roll_data.actuator[roll_mask]

            if len(roll_ff) > 0:
                ff_nonzero = np.abs(roll_ff) > 0.01
                if np.any(ff_nonzero):
                    physics.roll_ff_efficiency = np.mean(
                        np.abs(roll_actuator[ff_nonzero]) / np.abs(roll_ff[ff_nonzero])
                    )
                physics.roll_d_limit = float(np.min(roll_d))
                physics.roll_final_ff = float(roll_ff[-1])
                physics.roll_final_p = float(roll_p[-1])
                physics.roll_final_i = float(roll_i[-1])
                physics.roll_final_d = float(roll_d[-1])

        if np.any(pitch_mask):
            pitch_data = self.log_data.atrp
            pitch_ff = pitch_data.FF[pitch_mask]
            pitch_p = pitch_data.P[pitch_mask]
            pitch_i = pitch_data.I[pitch_mask]
            pitch_d = pitch_data.D[pitch_mask]
            pitch_actuator = pitch_data.actuator[pitch_mask]

            if len(pitch_ff) > 0:
                ff_nonzero = np.abs(pitch_ff) > 0.01
                if np.any(ff_nonzero):
                    physics.pitch_ff_efficiency = np.mean(
                        np.abs(pitch_actuator[ff_nonzero])
                        / np.abs(pitch_ff[ff_nonzero])
                    )
                physics.pitch_d_limit = float(np.min(pitch_d))
                physics.pitch_final_ff = float(pitch_ff[-1])
                physics.pitch_final_p = float(pitch_p[-1])
                physics.pitch_final_i = float(pitch_i[-1])
                physics.pitch_final_d = float(pitch_d[-1])

        return physics

    def get_targets_for_level(self, level: int) -> Tuple[float, float, float, float]:
        return self.AUTOTUNE_LEVEL_TARGETS.get(level, self.AUTOTUNE_LEVEL_TARGETS[7])

    def extract_data_for_phase(self, pid_data, phase: AnalysisPhase):
        """Extract PID data subset for a specific time phase"""
        if pid_data is None or pid_data.is_empty():
            return None

        mask = (pid_data.time >= phase.start_time) & (pid_data.time <= phase.end_time)

        from log_parser import PIDData

        return PIDData(
            time=pid_data.time[mask],
            target=pid_data.target[mask],
            actual=pid_data.actual[mask],
            error=pid_data.error[mask],
            P=pid_data.P[mask],
            I=pid_data.I[mask],
            D=pid_data.D[mask],
            FF=pid_data.FF[mask],
            DFF=pid_data.DFF[mask],
            Dmod=pid_data.Dmod[mask],
            SRate=pid_data.SRate[mask],
            Flags=pid_data.Flags[mask],
            sample_rate=pid_data.sample_rate,
        )

    def analyze_scaler_effects(
        self, pid_data, scaler_data, time_start: float, time_end: float
    ) -> Dict:
        """
        Analyze how speed scaler affects PID performance.

        At low airspeed (high scaler), PID gains are amplified, potentially causing twitchy behavior.
        At cruise airspeed (scaler ~1.0), gains are normal.
        At high airspeed (low scaler), gains are reduced, potentially causing sluggish response.

        Returns analysis of performance vs scaler.
        """
        if pid_data is None or pid_data.is_empty():
            return {}

        if scaler_data is None or len(scaler_data.time) == 0:
            return {"scaler_available": False}

        # Get scaler for each PID sample
        from scipy import interpolate

        if len(scaler_data.time) > 2:
            scaler_interp = interpolate.interp1d(
                scaler_data.time, scaler_data.scaler, bounds_error=False, fill_value=1.0
            )
            scalers = scaler_interp(pid_data.time)
        else:
            scalers = np.ones(len(pid_data.time))

        # Calculate error magnitude for each sample
        errors = np.abs(pid_data.error)

        # Group by scaler ranges
        results = {
            "scaler_available": True,
            "scaler_range": (float(np.min(scalers)), float(np.max(scalers))),
            "scaler_mean": float(np.mean(scalers)),
            "scaler_std": float(np.std(scalers)),
        }

        # Define scaler bands
        # Low speed (scaler > 1.2): Twitchy expected
        # Cruise (scaler 0.9-1.1): Should be smooth
        # High speed (scaler < 0.8): Sluggish expected

        low_speed_mask = scalers > 1.2
        cruise_mask = (scalers >= 0.9) & (scalers <= 1.1)
        high_speed_mask = scalers < 0.8

        if np.any(low_speed_mask):
            results["low_speed"] = {
                "scaler_range": (
                    float(np.min(scalers[low_speed_mask])),
                    float(np.max(scalers[low_speed_mask])),
                ),
                "mean_error": float(np.mean(errors[low_speed_mask])),
                "max_error": float(np.max(errors[low_speed_mask])),
                "sample_count": int(np.sum(low_speed_mask)),
                "expected_behavior": "twitchy",
            }

        if np.any(cruise_mask):
            results["cruise"] = {
                "scaler_range": (
                    float(np.min(scalers[cruise_mask])),
                    float(np.max(scalers[cruise_mask])),
                ),
                "mean_error": float(np.mean(errors[cruise_mask])),
                "max_error": float(np.max(errors[cruise_mask])),
                "sample_count": int(np.sum(cruise_mask)),
                "expected_behavior": "smooth",
            }

        if np.any(high_speed_mask):
            results["high_speed"] = {
                "scaler_range": (
                    float(np.min(scalers[high_speed_mask])),
                    float(np.max(scalers[high_speed_mask])),
                ),
                "mean_error": float(np.mean(errors[high_speed_mask])),
                "max_error": float(np.max(errors[high_speed_mask])),
                "sample_count": int(np.sum(high_speed_mask)),
                "expected_behavior": "sluggish",
            }

        # Calculate correlation between scaler and error
        if len(scalers) > 10:
            correlation = np.corrcoef(scalers, errors)[0, 1]
            results["scaler_error_correlation"] = float(correlation)

            if correlation > 0.3:
                results["analysis"] = (
                    "Higher error at low airspeed (high scaler). PID gains may be too aggressive for slow flight."
                )
            elif correlation < -0.3:
                results["analysis"] = (
                    "Higher error at high airspeed (low scaler). PID gains may be too weak for fast flight."
                )
            else:
                results["analysis"] = (
                    "Error relatively independent of airspeed. Good scaling behavior."
                )

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python flight_analyzer.py <logfile.bin>")
        sys.exit(1)

    from log_parser import LogParser

    log_data = LogParser(sys.argv[1]).parse()
    analyzer = FlightAnalyzer(log_data)

    # Show autotune sessions
    sessions = analyzer.get_autotune_sessions()
    if sessions:
        print(f"\nAutotune sessions found: {len(sessions)}")
        for s in sessions:
            print(
                f"  Session {s.index}: [{s.start_time:.1f}s - {s.end_time:.1f}s] ({s.duration_s:.0f}s)"
            )
            if s.has_roll:
                print(f"    Roll finished at {s.roll_finished_time:.1f}s")
            if s.has_pitch:
                print(f"    Pitch finished at {s.pitch_finished_time:.1f}s")

        print(
            f"\nSelect session to analyze (1-{len(sessions)}), or press Enter for first:"
        )
        # In non-interactive mode, default to first
        session_idx = 1
        print(f"Using session {session_idx}")
    else:
        session_idx = None
        print("No autotune sessions found, analyzing all suitable flight modes")

    phases, physics = analyzer.analyze(session_idx)

    print(f"\nAnalysis phases: {len(phases)}")
    for p in phases:
        print(
            f"  [{p.start_time:.1f}s - {p.end_time:.1f}s] {p.mode_name} "
            f"(alt={p.min_altitude_m:.0f}m, scaler={p.avg_scaler:.2f})"
        )
