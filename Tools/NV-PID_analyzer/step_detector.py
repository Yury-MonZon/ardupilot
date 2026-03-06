#!/usr/bin/env python3
"""
Step Detector - Robust step detection and response metrics calculation
Uses convolution methods for noise handling, adaptive to sample rate
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from log_parser import PIDData


@dataclass
class Step:
    """Detected step event"""

    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    initial_value: float
    target_value: float
    step_size: float
    direction: str  # 'up' or 'down'


@dataclass
class StepMetrics:
    """Calculated metrics for a step response"""

    rise_time: float  # Time from 10% to 90% of step
    settling_time: float  # Time to stay within ±5% of target
    overshoot_pct: float  # Maximum overshoot percentage
    steady_state_error: float  # Mean error in settled region
    peak_error: float  # Maximum error during step
    phase_margin_ok: bool  # Is phase margin acceptable


class StepDetector:
    """
    Robust step detection using derivative analysis and convolution smoothing.
    Adaptive to different sample rates and handles noisy real-flight data.
    """

    def __init__(self, pid_data: PIDData):
        self.data = pid_data
        self.sample_rate = self._detect_sample_rate()

    def _detect_sample_rate(self) -> float:
        """Auto-detect sample rate from time array"""
        if len(self.data.time) < 2:
            return 10.0  # Default

        dt = np.diff(self.data.time)
        # Use median for robustness to gaps
        median_dt = np.median(dt)
        if median_dt > 0:
            return 1.0 / median_dt
        return 10.0

    def smooth_gaussian(
        self, data: np.ndarray, sigma_seconds: float = 0.03
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing via convolution.
        Kernel size adapts to sample rate.
        """
        sigma_samples = sigma_seconds * self.sample_rate

        # Ensure minimum kernel size
        kernel_size = max(int(6 * sigma_samples), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = np.linspace(-3, 3, kernel_size)
        kernel = np.exp(-(x**2) / (2 * (sigma_samples / max(self.sample_rate, 1)) ** 2))
        kernel = kernel / kernel.sum()

        # Apply convolution
        smoothed = np.convolve(data, kernel, mode="same")
        return smoothed

    def smooth_savitzky_golay(
        self, data: np.ndarray, window_seconds: float = 0.1
    ) -> np.ndarray:
        """
        Savitzky-Golay filter for preserving peaks while smoothing.
        Good for step detection in noisy data.
        """
        window_samples = int(window_seconds * self.sample_rate)
        # Ensure odd window size and minimum of 5
        window_samples = max(window_samples | 1, 5)

        if len(data) < window_samples:
            return data

        # Polyorder must be less than window size
        polyorder = min(3, window_samples - 1)

        try:
            smoothed = signal.savgol_filter(data, window_samples, polyorder)
            return smoothed
        except Exception:
            return data

    def detect_steps(
        self,
        min_step_deg_s: float = 10.0,
        min_duration_s: float = 0.2,
        max_duration_s: float = 3.0,
    ) -> List[Step]:
        """
        Detect steps in the target rate signal.

        Algorithm:
        1. Smooth target signal to reduce noise
        2. Calculate derivative of smoothed signal
        3. Find peaks in derivative (step starts)
        4. Group consecutive peaks into step events
        5. Filter by minimum step size and duration

        Parameters:
            min_step_deg_s: Minimum step size in deg/s
            min_duration_s: Minimum step transition duration
            max_duration_s: Maximum step transition duration

        Returns:
            List of detected Step objects
        """
        if self.data.is_empty() or len(self.data.time) < 10:
            return []

        target = self.data.target
        time = self.data.time

        # Smooth the target signal (Gaussian + Savitzky-Golay cascade)
        target_smooth = self.smooth_gaussian(target, sigma_seconds=0.02)
        target_smooth = self.smooth_savitzky_golay(target_smooth, window_seconds=0.05)

        # Calculate derivative
        dt = np.diff(time)
        dt[dt == 0] = 1e-6  # Avoid division by zero
        d_target = np.diff(target_smooth) / dt

        # Smooth derivative
        d_target_smooth = self.smooth_gaussian(d_target, sigma_seconds=0.03)

        # Calculate adaptive threshold based on signal statistics
        noise_level = np.std(d_target_smooth)
        threshold = max(min_step_deg_s / min_duration_s, noise_level * 4)

        # Find step starts using hysteresis
        steps = []
        in_step = False
        step_start_idx = 0

        for i in range(len(d_target_smooth)):
            d = abs(d_target_smooth[i])

            if d > threshold and not in_step:
                # Potential step start
                step_start_idx = i
                in_step = True
            elif d < threshold * 0.3 and in_step:
                # Step transition complete
                step_end_idx = i

                # Validate step
                step = self._validate_step(
                    step_start_idx,
                    step_end_idx,
                    target,
                    target_smooth,
                    time,
                    min_step_deg_s,
                    min_duration_s,
                    max_duration_s,
                )

                if step is not None:
                    steps.append(step)

                in_step = False

        # Merge nearby steps if they're too close
        steps = self._merge_nearby_steps(steps, time, min_duration_s)

        return steps

    def _validate_step(
        self,
        start_idx: int,
        end_idx: int,
        target: np.ndarray,
        target_smooth: np.ndarray,
        time: np.ndarray,
        min_step: float,
        min_dur: float,
        max_dur: float,
    ) -> Optional[Step]:
        """Validate if detected step is real and meets criteria"""

        # Check duration
        duration = time[end_idx] - time[start_idx]
        if duration < min_dur or duration > max_dur:
            return None

        # Calculate step size
        # Use pre-step average and post-step average
        pre_start = max(0, start_idx - 5)
        post_end = min(len(target_smooth), end_idx + 10)

        initial_value = np.mean(target_smooth[pre_start : start_idx + 1])
        final_value = np.mean(target_smooth[end_idx:post_end])
        step_size = final_value - initial_value

        # Check step size
        if abs(step_size) < min_step:
            return None

        direction = "up" if step_size > 0 else "down"

        return Step(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=time[start_idx],
            end_time=time[end_idx],
            initial_value=initial_value,
            target_value=final_value,
            step_size=step_size,
            direction=direction,
        )

    def _merge_nearby_steps(
        self, steps: List[Step], time: np.ndarray, min_gap: float
    ) -> List[Step]:
        """Merge steps that are too close together"""
        if len(steps) <= 1:
            return steps

        merged = []
        current = steps[0]

        for next_step in steps[1:]:
            gap = next_step.start_time - current.end_time

            if gap < min_gap:
                # Merge: use the larger step's properties
                if abs(next_step.step_size) > abs(current.step_size):
                    current = next_step
            else:
                merged.append(current)
                current = next_step

        merged.append(current)
        return merged

    def calculate_metrics(self, step: Step) -> Optional[StepMetrics]:
        """
        Calculate step response metrics from detected step.

        Metrics:
        - Rise time: Time from 10% to 90% of step
        - Settling time: Time to stay within ±5% of target
        - Overshoot: Maximum percentage above target
        - Steady-state error: Mean error in settled region
        """

        # Extract step region with margin
        margin = max(0.5, step.step_size * 0.2)  # Time margin around step
        start_time = step.start_time - margin
        end_time = step.end_time + margin * 3

        mask = (self.data.time >= start_time) & (self.data.time <= end_time)

        if np.sum(mask) < 10:
            return None

        t = self.data.time[mask]
        actual = self.data.actual[mask]
        target = self.data.target[mask]

        # Smooth actual signal for cleaner metrics
        actual_smooth = self.smooth_gaussian(actual, sigma_seconds=0.02)
        target_smooth = self.smooth_gaussian(target, sigma_seconds=0.02)

        # Step size and direction
        step_size = step.step_size
        abs_step = abs(step_size)

        if abs_step < 1.0:
            return None

        # Normalize to [0, 1] range
        initial = actual_smooth[0] if len(actual_smooth) > 0 else actual[0]
        target_val = step.target_value

        # Calculate normalized response
        normalized = (actual_smooth - initial) / step_size

        # Find zero crossing (response start)
        zero_crossing_idx = np.argmax(np.abs(normalized) > 0.01)
        if zero_crossing_idx == 0:
            zero_crossing_idx = 1

        # Rise time: 10% to 90%
        try:
            idx_10 = zero_crossing_idx + np.argmax(
                np.abs(normalized[zero_crossing_idx:]) >= 0.1 * np.sign(step_size)
            )
            idx_90 = zero_crossing_idx + np.argmax(
                np.abs(normalized[zero_crossing_idx:]) >= 0.9 * np.sign(step_size)
            )

            if idx_10 >= len(t) or idx_90 >= len(t):
                return None

            rise_time = t[idx_90] - t[idx_10]
        except Exception:
            rise_time = float("nan")

        # Overshoot
        try:
            if step.direction == "up":
                peak = np.max(normalized)
                overshoot_pct = max(0, (peak - 1.0) * 100)
            else:
                peak = np.min(normalized)
                overshoot_pct = max(0, (-1.0 - peak) * 100)
        except Exception:
            overshoot_pct = 0.0

        # Settling time (within 5% of target)
        try:
            error = np.abs(actual_smooth - target_val)
            settling_threshold = 0.05 * abs_step

            # Find first point that stays within threshold
            settled_idx = len(t)
            for i in range(len(error) - 1, -1, -1):
                if error[i] > settling_threshold:
                    settled_idx = i + 1
                    break

            settling_time = (
                t[settled_idx] - t[0] if settled_idx < len(t) else float("inf")
            )
        except Exception:
            settling_time = float("nan")

        # Steady-state error (last 20% of data)
        try:
            last_20_start = int(0.8 * len(actual_smooth))
            steady_error = np.mean(actual_smooth[last_20_start:] - target_val)
        except Exception:
            steady_error = float("nan")

        # Peak error during step
        try:
            peak_error = np.max(np.abs(actual_smooth - target_smooth))
        except Exception:
            peak_error = float("nan")

        # Phase margin assessment (rough heuristic)
        phase_margin_ok = overshoot_pct < 25 and rise_time < 0.5

        return StepMetrics(
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot_pct=overshoot_pct,
            steady_state_error=steady_error,
            peak_error=peak_error,
            phase_margin_ok=phase_margin_ok,
        )

    def assess_step_quality(self, steps: List[Step]) -> Tuple[int, str]:
        """
        Assess quality of detected steps for step response analysis.

        Returns:
            (quality_score, message) where:
            - quality_score: 0-100, higher = better quality for analysis
            - message: description of quality assessment
        """
        if not steps:
            return (0, "No steps detected")

        metrics_list = []
        valid_steps = 0
        invalid_rise_time = 0
        total_steps = len(steps)

        for step in steps:
            metrics = self.calculate_metrics(step)
            if metrics is None:
                continue

            metrics_list.append(metrics)

            # Check for valid rise time (must be positive and reasonable)
            if (
                np.isnan(metrics.rise_time)
                or metrics.rise_time <= 0
                or metrics.rise_time > 5.0
            ):
                invalid_rise_time += 1
            else:
                valid_steps += 1

        if not metrics_list:
            return (0, "No valid step metrics calculated")

        # Calculate quality score
        valid_ratio = valid_steps / total_steps if total_steps > 0 else 0

        # Need at least 3 valid steps for meaningful average
        if valid_steps < 3:
            return (
                20,
                f"Only {valid_steps} valid steps (need >=3 for reliable metrics)",
            )

        # Check rise time consistency
        rise_times = [
            m.rise_time
            for m in metrics_list
            if not np.isnan(m.rise_time) and m.rise_time > 0
        ]
        if len(rise_times) >= 3:
            rise_time_std = np.std(rise_times)
            rise_time_mean = np.mean(rise_times)
            cv = rise_time_std / rise_time_mean if rise_time_mean > 0 else 999
        else:
            cv = 999

        # Quality assessment
        if valid_ratio < 0.3:
            return (
                30,
                f"Low valid step ratio ({valid_ratio * 100:.0f}%). Step metrics may be unreliable.",
            )

        if cv > 1.0:
            return (
                40,
                f"High rise time variability (CV={cv:.1f}). Steps may not be clean step inputs.",
            )

        if valid_steps >= 5 and valid_ratio >= 0.5:
            return (
                80,
                f"Good step quality: {valid_steps} valid steps, consistent rise times.",
            )

        if valid_steps >= 3:
            return (60, f"Acceptable step quality: {valid_steps} valid steps.")

        return (40, f"Limited step quality: {valid_steps} valid steps.")

    def calculate_average_metrics(self, steps: List[Step]) -> Optional[StepMetrics]:
        """Calculate average metrics across all detected steps"""
        if not steps:
            return None

        metrics_list = []
        for step in steps:
            metrics = self.calculate_metrics(step)
            if metrics is not None:
                metrics_list.append(metrics)

        if not metrics_list:
            return None

        # Calculate averages
        return StepMetrics(
            rise_time=np.nanmean([m.rise_time for m in metrics_list]),
            settling_time=np.nanmean([m.settling_time for m in metrics_list]),
            overshoot_pct=np.nanmean([m.overshoot_pct for m in metrics_list]),
            steady_state_error=np.nanmean([m.steady_state_error for m in metrics_list]),
            peak_error=np.nanmean([m.peak_error for m in metrics_list]),
            phase_margin_ok=all(m.phase_margin_ok for m in metrics_list),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step_detector.py <logfile.bin>")
        sys.exit(1)

    from log_parser import LogParser
    from flight_analyzer import FlightAnalyzer

    log_data = LogParser(sys.argv[1]).parse()
    analyzer = FlightAnalyzer(log_data)
    phases, _ = analyzer.analyze()

    print(f"\nStep Detection Results:")
    for phase in phases:
        print(
            f"\nPhase: {phase.mode_name} ({phase.start_time:.1f}s - {phase.end_time:.1f}s)"
        )

        # Extract roll data for this phase
        roll_phase = analyzer.extract_data_for_phase(log_data.roll, phase)
        if roll_phase and not roll_phase.is_empty():
            detector = StepDetector(roll_phase)
            steps = detector.detect_steps()
            print(f"  Roll: {len(steps)} steps detected")

            if steps:
                avg_metrics = detector.calculate_average_metrics(steps)
                if avg_metrics:
                    print(f"    Avg rise time: {avg_metrics.rise_time:.3f}s")
                    print(f"    Avg overshoot: {avg_metrics.overshoot_pct:.1f}%")
                    print(
                        f"    Avg steady error: {avg_metrics.steady_state_error:.2f} deg/s"
                    )
