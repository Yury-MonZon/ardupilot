#!/usr/bin/env python3
"""
Tuning Suggester - Generate PID tuning recommendations based on analysis

IMPORTANT: Dmod is NOT a pure D-term indicator!

Dmod is a slew rate limiter modifier for P+D combined output:
- Dmod = slew_limiter.modifier((P + D) * slew_limit_scale)
- Dmod < 1.0 means P+D output is exceeding actuator physical slew rate
- This causes actuator demand and achieved rate to get out of phase

Dmod < 1.0 could indicate:
- High P gain (P dominates)
- High D gain (D dominates)
- Aggressive maneuvering (large target changes)
- Actuator slew rate too slow for demanded rate

DO NOT blindly reduce D when Dmod < 1.0!
Analyze P and D individually to understand which is the cause.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from flight_analyzer import FlightAnalyzer, AutotunePhysics, AnalysisPhase
from pid_metrics import PIDMetricsCalculator, TrackingMetrics, OscillationMetrics
from step_detector import StepDetector, StepMetrics


@dataclass
class Suggestion:
    """A tuning suggestion"""

    priority: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    axis: str  # 'Roll' or 'Pitch'
    issue: str  # Description of the problem
    action: str  # What to do
    current_value: float  # Current parameter value
    suggested_value: float  # Suggested parameter value
    param_name: str  # Parameter name (e.g., 'RLL_RATE_D')
    reason: str  # Why this change is recommended


@dataclass
class PDAnalysis:
    """P/D balance analysis results"""

    p_mean: float
    d_mean: float
    p_max: float
    d_max: float
    pd_ratio: float
    p_dominant: bool
    d_dominant: bool
    assessment: str


class TuningSuggester:
    """
    Generate PID tuning suggestions based on analysis results.

    Uses proper understanding of ArduPilot PID:
    - Dmod < 1.0 indicates P+D slew rate limiting, NOT pure D oscillation
    - FF should do most of the work (50-70% contribution)
    - P/D ratio should be balanced (3-5 typical for fixed-wing)
    """

    PARAM_NAMES = {
        "roll": {
            "P": "RLL_RATE_P",
            "I": "RLL_RATE_I",
            "D": "RLL_RATE_D",
            "FF": "RLL_RATE_FF",
            "IMAX": "RLL_RATE_IMAX",
            "FLTD": "RLL_RATE_FLTD",
            "FLTT": "RLL_RATE_FLTT",
            "SMAX": "RLL_RATE_SMAX",
        },
        "pitch": {
            "P": "PTCH_RATE_P",
            "I": "PTCH_RATE_I",
            "D": "PTCH_RATE_D",
            "FF": "PTCH_RATE_FF",
            "IMAX": "PTCH_RATE_IMAX",
            "FLTD": "PTCH_RATE_FLTD",
            "FLTT": "PTCH_RATE_FLTT",
            "SMAX": "PTCH_RATE_SMAX",
        },
    }

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

    def __init__(
        self,
        params: Dict[str, float],
        autotune_level: int = 7,
        autotune_physics: Optional[AutotunePhysics] = None,
    ):
        self.params = params
        self.autotune_level = autotune_level
        self.autotune_physics = autotune_physics
        self.targets = self.AUTOTUNE_LEVEL_TARGETS.get(
            autotune_level, self.AUTOTUNE_LEVEL_TARGETS[7]
        )

    def analyze_pd_balance(
        self, P: np.ndarray, D: np.ndarray, target: np.ndarray, dmod: np.ndarray
    ) -> PDAnalysis:
        """
        Analyze P/D balance to understand which term dominates.

        This helps identify whether Dmod < 1.0 is caused by:
        - High P (P dominant)
        - High D (D dominant)
        - Both (both need reduction)
        - Neither (aggressive maneuvering)
        """
        # Analyze during active control
        active_mask = np.abs(target) > 3.0
        if not np.any(active_mask):
            active_mask = np.abs(P) > 0.01

        P_active = P[active_mask]
        D_active = D[active_mask]
        target_active = target[active_mask]
        dmod_active = dmod[active_mask]

        P_mean = np.mean(np.abs(P_active))
        D_mean = np.mean(np.abs(D_active))
        P_max = np.max(np.abs(P_active))
        D_max = np.max(np.abs(D_active))

        # P/D ratio
        pd_ratio = P_mean / D_mean if D_mean > 1e-6 else float("inf")

        # Analyze during low Dmod events specifically
        low_dmod_mask = dmod_active < 0.9
        if np.any(low_dmod_mask):
            P_low_dmod = np.mean(np.abs(P_active[low_dmod_mask]))
            D_low_dmod = np.mean(np.abs(D_active[low_dmod_mask]))
            target_low_dmod = np.mean(np.abs(target_active[low_dmod_mask]))

            # Compare to normal operation
            normal_mask = dmod_active >= 0.9
            if np.any(normal_mask):
                P_normal = np.mean(np.abs(P_active[normal_mask]))
                D_normal = np.mean(np.abs(D_active[normal_mask]))

                # Which term increased more during low Dmod?
                p_increase = P_low_dmod / P_normal if P_normal > 0 else 1.0
                d_increase = D_low_dmod / D_normal if D_normal > 0 else 1.0

                p_dominant = p_increase > d_increase * 1.5
                d_dominant = d_increase > p_increase * 1.5
            else:
                p_dominant = P_low_dmod > D_low_dmod * 2
                d_dominant = D_low_dmod > P_low_dmod * 2
        else:
            p_dominant = False
            d_dominant = False

        # Assessment
        if pd_ratio > 8:
            assessment = "high_p_ratio"
        elif pd_ratio < 1.5:
            assessment = "low_p_ratio"
        elif p_dominant:
            assessment = "p_dominant_during_slew_limit"
        elif d_dominant:
            assessment = "d_dominant_during_slew_limit"
        else:
            assessment = "balanced"

        return PDAnalysis(
            p_mean=P_mean,
            d_mean=D_mean,
            p_max=P_max,
            d_max=D_max,
            pd_ratio=pd_ratio,
            p_dominant=p_dominant,
            d_dominant=d_dominant,
            assessment=assessment,
        )

    def suggest(
        self,
        axis: str,
        tracking: TrackingMetrics,
        oscillation: OscillationMetrics,
        step_metrics: Optional[StepMetrics],
        ff_contribution: float,
        pd_analysis: Optional[PDAnalysis] = None,
        pid_data: Optional[Dict] = None,
        step_quality_score: int = 100,
    ) -> List[Suggestion]:
        """
        Generate tuning suggestions for one axis.

        IMPORTANT: Dmod < 1.0 is NOT a D-term problem alone!
        It indicates P+D combined output exceeds actuator slew rate.

        step_quality_score: 0-100, higher = better step quality.
                            If < 50, overshoot-based suggestions are de-emphasized.
        """
        suggestions = []
        axis_lower = axis.lower()
        param_names = self.PARAM_NAMES.get(axis_lower, {})
        axis_upper = axis.upper()

        target_overshoot, target_rise_time = self.targets[2], self.targets[3]

        current_p = self.params.get(param_names.get("P", ""), 0.1)
        current_i = self.params.get(param_names.get("I", ""), 0.1)
        current_d = self.params.get(param_names.get("D", ""), 0.01)
        current_ff = self.params.get(param_names.get("FF", ""), 0.2)
        current_smax = self.params.get(param_names.get("SMAX", ""), 150.0)

        # Rule 1: Slew rate limiting detected (Dmod < 1.0)
        # CRITICAL: This is P+D problem, NOT just D!
        if oscillation and oscillation.oscillation_detected:
            if pd_analysis:
                if pd_analysis.p_dominant:
                    # P is causing the slew limiting
                    suggestions.append(
                        Suggestion(
                            priority="CRITICAL",
                            axis=axis,
                            issue=f"P+D slew rate limiting (Dmod={oscillation.min_dmod:.2f}), P dominant",
                            action="Decrease P gain",
                            current_value=current_p,
                            suggested_value=round(current_p * 0.75, 4),
                            param_name=param_names.get("P", f"{axis_upper}_RATE_P"),
                            reason="P output is dominant during slew limiting. Dmod < 1.0 means P+D exceeds actuator rate.",
                        )
                    )
                elif pd_analysis.d_dominant:
                    # D is causing the slew limiting
                    suggestions.append(
                        Suggestion(
                            priority="CRITICAL",
                            axis=axis,
                            issue=f"P+D slew rate limiting (Dmod={oscillation.min_dmod:.2f}), D dominant",
                            action="Decrease D gain",
                            current_value=current_d,
                            suggested_value=round(current_d * 0.7, 5),
                            param_name=param_names.get("D", f"{axis_upper}_RATE_D"),
                            reason="D output is dominant during slew limiting. Dmod < 1.0 means P+D exceeds actuator rate.",
                        )
                    )
                else:
                    # Both are contributing, or aggressive maneuvering
                    suggestions.append(
                        Suggestion(
                            priority="HIGH",
                            axis=axis,
                            issue=f"P+D slew rate limiting (Dmod={oscillation.min_dmod:.2f}), balanced",
                            action="Check SMAX or reduce P slightly",
                            current_value=current_p,
                            suggested_value=round(current_p * 0.85, 4),
                            param_name=param_names.get("P", f"{axis_upper}_RATE_P"),
                            reason="Both P and D contribute to slew limiting. Could also increase SMAX if actuator can handle it.",
                        )
                    )
            else:
                # No PD analysis available, give general warning
                suggestions.append(
                    Suggestion(
                        priority="INFO",
                        axis=axis,
                        issue=f"P+D slew rate limiting detected (Dmod={oscillation.min_dmod:.2f})",
                        action="Analyze P/D balance to determine cause",
                        current_value=0,
                        suggested_value=0,
                        param_name="N/A",
                        reason="Dmod < 1.0 means P+D output exceeds actuator slew rate. Check which term dominates.",
                    )
                )

        # Rule 2: P/D ratio assessment
        if pd_analysis:
            if pd_analysis.assessment == "high_p_ratio":
                suggestions.append(
                    Suggestion(
                        priority="LOW",
                        axis=axis,
                        issue=f"High P/D ratio ({pd_analysis.pd_ratio:.1f}:1)",
                        action="Consider increasing D for better damping",
                        current_value=current_d,
                        suggested_value=round(current_d * 1.3, 5),
                        param_name=param_names.get("D", f"{axis_upper}_RATE_D"),
                        reason=f"P/D ratio of {pd_analysis.pd_ratio:.1f} is high. Typical is 3-5 for fixed-wing.",
                    )
                )
            elif pd_analysis.assessment == "low_p_ratio":
                suggestions.append(
                    Suggestion(
                        priority="LOW",
                        axis=axis,
                        issue=f"Low P/D ratio ({pd_analysis.pd_ratio:.1f}:1)",
                        action="Consider reducing D or increasing P",
                        current_value=current_d,
                        suggested_value=round(current_d * 0.8, 5),
                        param_name=param_names.get("D", f"{axis_upper}_RATE_D"),
                        reason=f"P/D ratio of {pd_analysis.pd_ratio:.1f} is low. D might cause overshoot damping.",
                    )
                )

        # Rule 3: High overshoot (only if step quality is good)
        if step_metrics and step_metrics.overshoot_pct > target_overshoot * 1.5:
            # De-emphasize overshoot suggestions if step quality is poor
            if step_quality_score < 50:
                suggestions.append(
                    Suggestion(
                        priority="INFO",
                        axis=axis,
                        issue=f"High overshoot detected ({step_metrics.overshoot_pct:.1f}%)",
                        action="Verify with tracking quality metrics",
                        current_value=0,
                        suggested_value=0,
                        param_name="N/A",
                        reason=f"Step quality is poor (score={step_quality_score}). Overshoot metric may not be reliable. Check RMS error and Dmod instead.",
                    )
                )
            else:
                overshoot_factor = step_metrics.overshoot_pct / target_overshoot

                # For overshoot, check if it's P or lack of D
                if pd_analysis and pd_analysis.pd_ratio > 6:
                    # High P relative to D
                    suggested_p = current_p * (0.7 if overshoot_factor > 2 else 0.85)
                    suggestions.append(
                        Suggestion(
                            priority="HIGH" if step_quality_score >= 70 else "MEDIUM",
                            axis=axis,
                            issue=f"High overshoot ({step_metrics.overshoot_pct:.1f}%)",
                            action="Decrease P gain",
                            current_value=current_p,
                            suggested_value=round(suggested_p, 4),
                            param_name=param_names.get("P", f"{axis_upper}_RATE_P"),
                            reason=f"Overshoot {step_metrics.overshoot_pct:.1f}% with high P/D ratio suggests P is too high",
                        )
                    )
                else:
                    # Could benefit from more D
                    suggested_d = current_d * 1.3
                    suggestions.append(
                        Suggestion(
                            priority="HIGH" if step_quality_score >= 70 else "MEDIUM",
                            axis=axis,
                            issue=f"High overshoot ({step_metrics.overshoot_pct:.1f}%)",
                            action="Increase D gain for damping",
                            current_value=current_d,
                            suggested_value=round(suggested_d, 5),
                            param_name=param_names.get("D", f"{axis_upper}_RATE_D"),
                            reason=f"Overshoot {step_metrics.overshoot_pct:.1f}% could be reduced with more D damping",
                        )
                    )

        # Rule 4: Slow response
        if step_metrics and not np.isnan(step_metrics.rise_time):
            if step_metrics.rise_time > target_rise_time * 1.5:
                if ff_contribution < 40:
                    suggested_ff = current_ff * 1.15
                    suggestions.append(
                        Suggestion(
                            priority="MEDIUM",
                            axis=axis,
                            issue=f"Slow response (rise time={step_metrics.rise_time:.2f}s)",
                            action="Increase FF gain",
                            current_value=current_ff,
                            suggested_value=round(suggested_ff, 4),
                            param_name=param_names.get("FF", f"{axis_upper}_RATE_FF"),
                            reason="Low FF contribution with slow response. FF should do most work for good tracking.",
                        )
                    )
                else:
                    suggested_p = current_p * 1.15
                    suggestions.append(
                        Suggestion(
                            priority="MEDIUM",
                            axis=axis,
                            issue=f"Slow response (rise time={step_metrics.rise_time:.2f}s)",
                            action="Increase P gain",
                            current_value=current_p,
                            suggested_value=round(suggested_p, 4),
                            param_name=param_names.get("P", f"{axis_upper}_RATE_P"),
                            reason="FF contribution is good but response is slow. P might need increase.",
                        )
                    )

        # Rule 5: Steady-state error
        if step_metrics and not np.isnan(step_metrics.steady_state_error):
            if abs(step_metrics.steady_state_error) > 1.5:
                suggested_i = current_i * 1.2
                suggestions.append(
                    Suggestion(
                        priority="LOW",
                        axis=axis,
                        issue=f"Steady-state error ({step_metrics.steady_state_error:.2f} deg/s)",
                        action="Increase I gain",
                        current_value=current_i,
                        suggested_value=round(suggested_i, 4),
                        param_name=param_names.get("I", f"{axis_upper}_RATE_I"),
                        reason="I term should eliminate steady-state error over time",
                    )
                )

        # Rule 6: FF efficiency
        if ff_contribution < 30:
            suggestions.append(
                Suggestion(
                    priority="HIGH",
                    axis=axis,
                    issue=f"Low FF contribution ({ff_contribution:.1f}%)",
                    action="Increase FF gain",
                    current_value=current_ff,
                    suggested_value=round(current_ff * 1.2, 4),
                    param_name=param_names.get("FF", f"{axis_upper}_RATE_FF"),
                    reason=f"FF should do 50-70% of work. Currently only {ff_contribution:.1f}%.",
                )
            )
        elif ff_contribution > 75:
            suggestions.append(
                Suggestion(
                    priority="INFO",
                    axis=axis,
                    issue=f"High FF contribution ({ff_contribution:.1f}%)",
                    action="Check tracking quality",
                    current_value=current_ff,
                    suggested_value=current_ff,
                    param_name=param_names.get("FF", f"{axis_upper}_RATE_FF"),
                    reason=f"FF doing {ff_contribution:.1f}% of work. Good for fixed-wing if tracking is good.",
                )
            )

        # Rule 7: Good tuning
        if len(suggestions) == 0:
            suggestions.append(
                Suggestion(
                    priority="INFO",
                    axis=axis,
                    issue="No issues detected",
                    action="No changes needed",
                    current_value=0,
                    suggested_value=0,
                    param_name="N/A",
                    reason=f"All metrics within targets for AUTOTUNE_LEVEL {self.autotune_level}",
                )
            )

        return suggestions

    def format_suggestions(self, suggestions: List[Suggestion]) -> str:
        """Format suggestions for console output"""
        lines = []
        priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

        for priority in priorities:
            matching = [s for s in suggestions if s.priority == priority]
            if not matching:
                continue

            for s in matching:
                if s.priority == "INFO":
                    lines.append(f"  ✓ {s.axis}: {s.reason}")
                else:
                    symbol = "⚠" if s.priority in ["CRITICAL", "HIGH"] else "○"
                    lines.append(f"  {symbol} [{s.priority}] {s.axis}: {s.issue}")
                    lines.append(f"      Action: {s.action}")
                    if s.param_name != "N/A":
                        lines.append(
                            f"      {s.param_name}: {s.current_value:.4f} → {s.suggested_value:.4f}"
                        )
                    lines.append(f"      Reason: {s.reason}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("Tuning Suggester module - use via nv_pid_analyzer.py")
