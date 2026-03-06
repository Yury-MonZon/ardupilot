#!/usr/bin/env python3
"""
Autotune Analyzer - Extract and analyze step responses from autotune sessions

Uses ATRP data which contains deliberate step inputs during autotune.
Filters out steps where PID parameters were being changed.
Compares different PID values to find optimal gains.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from log_parser import ATRPData, LogData

# Autotune states from AP_AutoTune.h
AT_STATE_IDLE = 0
AT_STATE_DEMAND_POS = 1  # Positive step input
AT_STATE_DEMAND_NEG = 2  # Negative step input

# Autotune actions from AP_AutoTune.h
AT_ACTION_NONE = 0
AT_ACTION_LOW_RATE = 1
AT_ACTION_SHORT = 2
AT_ACTION_RAISE_PD = 3
AT_ACTION_LOWER_PD = 4
AT_ACTION_IDLE_LOWER_PD = 5
AT_ACTION_RAISE_D = 6
AT_ACTION_RAISE_P = 7
AT_ACTION_LOWER_D = 8
AT_ACTION_LOWER_P = 9


@dataclass
class AutotuneStep:
    """A single step response from autotune"""

    start_time: float
    end_time: float
    axis: int  # 0=roll, 1=pitch
    direction: str  # 'up' or 'down'
    state: int  # ATState

    # PID gains during this step
    ff: float
    p: float
    i: float
    d: float

    # Response metrics
    actuator_target: float = 0.0
    actuator_achieved: float = 0.0
    rise_time: float = 0.0
    overshoot: float = 0.0
    settling_time: float = 0.0

    # Quality flags
    params_changed_during: bool = False
    is_good_step: bool = True


@dataclass
class PIDGains:
    """PID gain set"""

    ff: float
    p: float
    i: float
    d: float

    def __hash__(self):
        return hash(
            (round(self.ff, 4), round(self.p, 4), round(self.i, 4), round(self.d, 4))
        )

    def __eq__(self, other):
        if not isinstance(other, PIDGains):
            return False
        return (
            abs(self.ff - other.ff) < 0.0001
            and abs(self.p - other.p) < 0.0001
            and abs(self.i - other.i) < 0.0001
            and abs(self.d - other.d) < 0.0001
        )


@dataclass
class StepResponseMetrics:
    """Metrics for a step response"""

    rise_time: float
    overshoot: float
    settling_time: float
    steady_state_error: float
    quality_score: float  # 0-100


class AutotuneAnalyzer:
    """
    Analyze autotune sessions to extract clean step responses.

    Autotune deliberately creates step inputs, making it ideal for
    step response analysis. We filter out steps where gains were
    being changed.
    """

    MIN_STEP_DURATION_S = 0.3
    MAX_STEP_DURATION_S = 5.0

    def __init__(self, atrp_data: ATRPData):
        self.atrp = atrp_data
        self.steps: List[AutotuneStep] = []

    def extract_steps(self) -> List[AutotuneStep]:
        """
        Extract all step responses from autotune data.

        Step responses occur when state == DEMAND_POS or DEMAND_NEG.
        We only keep steps where gains remained constant.
        """
        if self.atrp is None or len(self.atrp.time) == 0:
            return []

        steps = []

        # Process each axis separately
        for axis in [0, 1]:
            axis_mask = self.atrp.axis == axis
            if not np.any(axis_mask):
                continue

            axis_times = self.atrp.time[axis_mask]
            axis_states = self.atrp.state[axis_mask]
            axis_ff = self.atrp.FF[axis_mask]
            axis_p = self.atrp.P[axis_mask]
            axis_i = self.atrp.I[axis_mask]
            axis_d = self.atrp.D[axis_mask]
            axis_actuator = self.atrp.actuator[axis_mask]

            # Find state transitions (entering DEMAND_POS or DEMAND_NEG)
            in_step = False
            step_start_idx = 0
            step_gains = None

            for i in range(len(axis_states)):
                state = axis_states[i]

                if state in [AT_STATE_DEMAND_POS, AT_STATE_DEMAND_NEG] and not in_step:
                    # Start of a step
                    in_step = True
                    step_start_idx = i
                    step_gains = PIDGains(
                        ff=axis_ff[i], p=axis_p[i], i=axis_i[i], d=axis_d[i]
                    )

                elif state == AT_STATE_IDLE and in_step:
                    # End of step
                    step_end_idx = i
                    duration = axis_times[step_end_idx] - axis_times[step_start_idx]

                    # Validate step
                    if self.MIN_STEP_DURATION_S <= duration <= self.MAX_STEP_DURATION_S:
                        # Check if gains changed during step
                        gains_changed = self._check_gains_changed(
                            axis_ff,
                            axis_p,
                            axis_i,
                            axis_d,
                            step_start_idx,
                            step_end_idx,
                        )

                        direction = (
                            "up"
                            if axis_states[step_start_idx] == AT_STATE_DEMAND_POS
                            else "down"
                        )

                        step = AutotuneStep(
                            start_time=axis_times[step_start_idx],
                            end_time=axis_times[step_end_idx],
                            axis=axis,
                            direction=direction,
                            state=axis_states[step_start_idx],
                            ff=step_gains.ff,
                            p=step_gains.p,
                            i=step_gains.i,
                            d=step_gains.d,
                            params_changed_during=gains_changed,
                            is_good_step=not gains_changed,
                        )
                        steps.append(step)

                    in_step = False

        self.steps = steps
        return steps

    def _check_gains_changed(self, ff, p, i, d, start_idx, end_idx) -> bool:
        """Check if gains changed during step"""
        ff_range = np.max(ff[start_idx : end_idx + 1]) - np.min(
            ff[start_idx : end_idx + 1]
        )
        p_range = np.max(p[start_idx : end_idx + 1]) - np.min(
            p[start_idx : end_idx + 1]
        )
        i_range = np.max(i[start_idx : end_idx + 1]) - np.min(
            i[start_idx : end_idx + 1]
        )
        d_range = np.max(d[start_idx : end_idx + 1]) - np.min(
            d[start_idx : end_idx + 1]
        )

        # Consider gains changed if any parameter changed by more than 1%
        return (
            ff_range > 0.001 or p_range > 0.001 or i_range > 0.001 or d_range > 0.0005
        )

    def get_good_steps(self, axis: int = None) -> List[AutotuneStep]:
        """Get steps where gains remained constant"""
        good = [s for s in self.steps if s.is_good_step]
        if axis is not None:
            good = [s for s in good if s.axis == axis]
        return good

    def calculate_step_metrics(
        self, step: AutotuneStep
    ) -> Optional[StepResponseMetrics]:
        """
        Calculate response metrics for a step.

        Uses actuator data from ATRP to measure response.
        """
        # Get data for this step
        axis_mask = self.atrp.axis == step.axis
        time_mask = (self.atrp.time >= step.start_time) & (
            self.atrp.time <= step.end_time
        )
        mask = axis_mask & time_mask

        if not np.any(mask):
            return None

        times = self.atrp.time[mask]
        actuator = self.atrp.actuator[mask]

        if len(times) < 5:
            return None

        # Calculate metrics
        initial_actuator = actuator[0]
        final_actuator = actuator[-1]
        step_size = final_actuator - initial_actuator

        if abs(step_size) < 0.01:
            return None

        # Normalize actuator to [0, 1]
        normalized = (actuator - initial_actuator) / step_size

        # Rise time (10% to 90%)
        try:
            if step.direction == "up":
                idx_10 = np.argmax(normalized >= 0.1)
                idx_90 = np.argmax(normalized >= 0.9)
            else:
                idx_10 = np.argmax(normalized <= -0.1)
                idx_90 = np.argmax(normalized <= -0.9)

            if idx_10 < idx_90 and idx_90 < len(times):
                rise_time = times[idx_90] - times[idx_10]
            else:
                rise_time = float("nan")
        except:
            rise_time = float("nan")

        # Overshoot
        try:
            if step.direction == "up":
                peak = np.max(normalized)
                overshoot = max(0, (peak - 1.0) * 100)
            else:
                peak = np.min(normalized)
                overshoot = max(0, (-1.0 - peak) * 100)
        except:
            overshoot = 0.0

        # Settling time (within 5% of final)
        try:
            error = (
                np.abs(normalized - 1.0)
                if step.direction == "up"
                else np.abs(normalized + 1.0)
            )
            settled_idx = len(times)
            for i in range(len(error) - 1, -1, -1):
                if error[i] > 0.05:
                    settled_idx = i + 1
                    break
            settling_time = times[min(settled_idx, len(times) - 1)] - times[0]
        except:
            settling_time = float("nan")

        # Quality score (based on reasonable metrics)
        quality_score = 100.0
        if rise_time > 1.0 or rise_time < 0.01:
            quality_score -= 30
        if overshoot > 50:
            quality_score -= 20
        if overshoot > 100:
            quality_score -= 30
        if settling_time > 2.0:
            quality_score -= 20

        return StepResponseMetrics(
            rise_time=rise_time,
            overshoot=overshoot,
            settling_time=settling_time,
            steady_state_error=0.0,  # Not easily calculated from ATRP
            quality_score=max(0, quality_score),
        )

    def group_by_gains(self, axis: int) -> Dict[PIDGains, List[AutotuneStep]]:
        """Group steps by PID gains to compare different tunings"""
        good_steps = self.get_good_steps(axis)

        groups = {}
        for step in good_steps:
            gains = PIDGains(ff=step.ff, p=step.p, i=step.i, d=step.d)
            if gains not in groups:
                groups[gains] = []
            groups[gains].append(step)

        return groups

    def find_optimal_gains(self, axis: int) -> Tuple[Optional[PIDGains], Dict]:
        """
        Find optimal PID gains based on step response analysis.

        Compares different gain sets used during autotune and
        identifies which produced the best response.
        """
        groups = self.group_by_gains(axis)

        if not groups:
            return None, {}

        # Calculate average metrics for each gain set
        gain_scores = {}

        for gains, steps in groups.items():
            if len(steps) < 2:
                continue  # Need at least 2 steps for reliable metrics

            metrics_list = []
            for step in steps:
                metrics = self.calculate_step_metrics(step)
                if metrics and metrics.quality_score > 0:
                    metrics_list.append(metrics)

            if not metrics_list:
                continue

            # Average metrics
            avg_rise = np.nanmean([m.rise_time for m in metrics_list])
            avg_overshoot = np.nanmean([m.overshoot for m in metrics_list])
            avg_quality = np.mean([m.quality_score for m in metrics_list])

            gain_scores[gains] = {
                "avg_rise_time": avg_rise,
                "avg_overshoot": avg_overshoot,
                "avg_quality": avg_quality,
                "num_steps": len(metrics_list),
                "metrics_list": metrics_list,
            }

        if not gain_scores:
            return None, {}

        # Find best gains (highest quality score, lowest overshoot)
        best_gains = max(
            gain_scores.keys(),
            key=lambda g: (
                gain_scores[g]["avg_quality"],
                -gain_scores[g]["avg_overshoot"],
            ),
        )

        return best_gains, gain_scores

    def interpolate_optimal_gains(self, axis: int) -> Optional[PIDGains]:
        """
        Interpolate to find optimal gains between tested values.

        Uses linear interpolation based on overshoot vs gain relationships.
        """
        best_gains, gain_scores = self.find_optimal_gains(axis)

        if not best_gains or len(gain_scores) < 2:
            return best_gains

        # Simple approach: if best gains have low overshoot, suggest slightly
        # higher D for better damping. If high overshoot, keep current.
        best_score = gain_scores[best_gains]

        if best_score["avg_overshoot"] > 20:
            # High overshoot - suggest slightly higher D
            return PIDGains(
                ff=best_gains.ff,
                p=best_gains.p * 0.95,  # Slightly lower P
                i=best_gains.i,
                d=best_gains.d * 1.1,  # Higher D
            )
        elif best_score["avg_overshoot"] < 5:
            # Very low overshoot - could increase P for faster response
            return PIDGains(
                ff=best_gains.ff,
                p=best_gains.p * 1.05,
                i=best_gains.i,
                d=best_gains.d,
            )

        return best_gains


if __name__ == "__main__":
    import sys
    from log_parser import LogParser

    if len(sys.argv) < 2:
        print("Usage: python autotune_analyzer.py <logfile.bin>")
        sys.exit(1)

    log = LogParser(sys.argv[1]).parse()

    if log.atrp is None:
        print("No ATRP data found in log")
        sys.exit(1)

    analyzer = AutotuneAnalyzer(log.atrp)
    steps = analyzer.extract_steps()

    print(f"\nAutotune Step Analysis")
    print(f"Total steps extracted: {len(steps)}")

    for axis_name, axis_num in [("Roll", 0), ("Pitch", 1)]:
        good_steps = analyzer.get_good_steps(axis_num)
        print(f"\n{axis_name}:")
        print(f"  Good steps: {len(good_steps)}")

        if good_steps:
            best_gains, scores = analyzer.find_optimal_gains(axis_num)
            if best_gains:
                print(f"  Best gains found:")
                print(
                    f"    FF={best_gains.ff:.4f} P={best_gains.p:.4f} I={best_gains.i:.4f} D={best_gains.d:.4f}"
                )
                if best_gains in scores:
                    s = scores[best_gains]
                    print(f"    Avg overshoot: {s['avg_overshoot']:.1f}%")
                    print(f"    Avg rise time: {s['avg_rise_time'] * 1000:.0f}ms")
                    print(f"    Quality score: {s['avg_quality']:.0f}/100")
