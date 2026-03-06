#!/usr/bin/env python3
"""
PID Metrics Calculator - Calculate tracking performance and oscillation analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from log_parser import PIDData


@dataclass
class TrackingMetrics:
    """Tracking performance metrics"""

    rms_error: float  # Root mean square error
    mean_error: float  # Mean error (bias)
    max_error: float  # Maximum absolute error
    std_error: float  # Standard deviation of error
    correlation: float  # Correlation between target and actual


@dataclass
class OscillationMetrics:
    """Oscillation analysis from Dmod"""

    min_dmod: float  # Minimum Dmod value
    mean_dmod: float  # Mean Dmod value
    oscillation_detected: bool  # True if Dmod < 1.0 detected
    oscillation_pct: float  # Percentage of time oscillating
    oscillation_severity: str  # 'none', 'minor', 'moderate', 'severe'


@dataclass
class PIDContributions:
    """PID component contribution analysis"""

    p_pct: float  # P contribution percentage
    i_pct: float  # I contribution percentage
    d_pct: float  # D contribution percentage
    ff_pct: float  # FF contribution percentage
    dff_pct: float  # DFF contribution percentage


class PIDMetricsCalculator:
    """Calculate PID performance metrics"""

    def calculate_tracking(self, pid_data: PIDData) -> Optional[TrackingMetrics]:
        """
        Calculate tracking performance metrics.

        Parameters:
            pid_data: PIDData object with target, actual, error arrays

        Returns:
            TrackingMetrics object
        """
        if pid_data.is_empty():
            return None

        target = pid_data.target
        actual = pid_data.actual
        error = pid_data.error

        # RMS error
        rms_error = np.sqrt(np.mean(error**2))

        # Mean error (bias)
        mean_error = np.mean(error)

        # Max error
        max_error = np.max(np.abs(error))

        # Standard deviation
        std_error = np.std(error)

        # Correlation coefficient
        try:
            if np.std(target) > 0 and np.std(actual) > 0:
                correlation = np.corrcoef(target, actual)[0, 1]
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0

        return TrackingMetrics(
            rms_error=rms_error,
            mean_error=mean_error,
            max_error=max_error,
            std_error=std_error,
            correlation=correlation,
        )

    def analyze_dmod(self, pid_data: PIDData) -> Optional[OscillationMetrics]:
        """
        Analyze Dmod for oscillation detection.

        Dmod < 1.0 indicates oscillation (slew rate limiting of P+D output)
        - Dmod = 1.0: No oscillation, full P+D gain available
        - Dmod < 1.0: Oscillation detected, P+D gain being reduced
        - Dmod < 0.9: Significant oscillation
        - Dmod < 0.5: Severe oscillation

        Parameters:
            pid_data: PIDData object with Dmod array

        Returns:
            OscillationMetrics object
        """
        if pid_data.is_empty():
            return None

        dmod = pid_data.Dmod

        # Basic statistics
        min_dmod = np.min(dmod)
        mean_dmod = np.mean(dmod)

        # Oscillation detection
        oscillation_detected = min_dmod < 1.0

        # Percentage of time oscillating
        oscillation_pct = np.sum(dmod < 1.0) / len(dmod) * 100

        # Severity classification
        if min_dmod >= 1.0:
            severity = "none"
        elif min_dmod >= 0.9:
            severity = "minor"
        elif min_dmod >= 0.7:
            severity = "moderate"
        else:
            severity = "severe"

        return OscillationMetrics(
            min_dmod=min_dmod,
            mean_dmod=mean_dmod,
            oscillation_detected=oscillation_detected,
            oscillation_pct=oscillation_pct,
            oscillation_severity=severity,
        )

    def analyze_contributions(self, pid_data: PIDData) -> Optional[PIDContributions]:
        """
        Analyze contribution of each PID component.

        The logged P, I, D, FF values are in degrees of deflection.
        This shows what percentage each component contributes to the total output.

        Parameters:
            pid_data: PIDData object with P, I, D, FF, DFF arrays

        Returns:
            PIDContributions object
        """
        if pid_data.is_empty():
            return None

        P = np.abs(pid_data.P)
        I = np.abs(pid_data.I)
        D = np.abs(pid_data.D)
        FF = np.abs(pid_data.FF)
        DFF = np.abs(pid_data.DFF)

        # Total magnitude
        total = P + I + D + FF + DFF

        # Avoid division by zero
        if np.sum(total) < 1e-6:
            return PIDContributions(0, 0, 0, 0, 0)

        # Calculate percentages (using means)
        mean_total = np.mean(total)

        p_pct = np.mean(P) / mean_total * 100 if mean_total > 0 else 0
        i_pct = np.mean(I) / mean_total * 100 if mean_total > 0 else 0
        d_pct = np.mean(D) / mean_total * 100 if mean_total > 0 else 0
        ff_pct = np.mean(FF) / mean_total * 100 if mean_total > 0 else 0
        dff_pct = np.mean(DFF) / mean_total * 100 if mean_total > 0 else 0

        return PIDContributions(
            p_pct=p_pct, i_pct=i_pct, d_pct=d_pct, ff_pct=ff_pct, dff_pct=dff_pct
        )

    def get_ff_efficiency(self, pid_data: PIDData) -> float:
        """
        Calculate FF efficiency: how well FF predicts the required output.

        High FF efficiency means the FF term does most of the work,
        which is good for aircraft response.

        Returns:
            FF efficiency ratio (0-1, higher is better)
        """
        if pid_data.is_empty():
            return 0.0

        # FF efficiency is the ratio of FF contribution to total
        contributions = self.analyze_contributions(pid_data)
        if contributions is None:
            return 0.0

        # Higher FF percentage indicates better FF tuning
        # For aircraft, FF should dominate (50%+)
        return contributions.ff_pct / 100.0

    def analyze_rate_tracking_quality(self, pid_data: PIDData) -> Tuple[str, float]:
        """
        Overall assessment of rate tracking quality.

        Returns:
            (quality_rating, score) tuple
            quality_rating: 'excellent', 'good', 'fair', 'poor'
            score: 0-100 score
        """
        if pid_data.is_empty():
            return "unknown", 0.0

        tracking = self.calculate_tracking(pid_data)
        oscillation = self.analyze_dmod(pid_data)
        contributions = self.analyze_contributions(pid_data)

        if tracking is None:
            return "unknown", 0.0

        score = 100.0

        # Deduct for high RMS error
        if tracking.rms_error > 5.0:
            score -= 30
        elif tracking.rms_error > 2.0:
            score -= 15
        elif tracking.rms_error > 1.0:
            score -= 5

        # Deduct for high max error
        if tracking.max_error > 20.0:
            score -= 20
        elif tracking.max_error > 10.0:
            score -= 10
        elif tracking.max_error > 5.0:
            score -= 5

        # Deduct for oscillation
        if oscillation:
            if oscillation.oscillation_severity == "severe":
                score -= 30
            elif oscillation.oscillation_severity == "moderate":
                score -= 20
            elif oscillation.oscillation_severity == "minor":
                score -= 10

        # Deduct for poor FF contribution (aircraft should have good FF)
        if contributions:
            if contributions.ff_pct < 30:
                score -= 15
            elif contributions.ff_pct < 40:
                score -= 5

        # Determine rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 50:
            rating = "fair"
        else:
            rating = "poor"

        return rating, max(0, score)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pid_metrics.py <logfile.bin>")
        sys.exit(1)

    from log_parser import LogParser
    from flight_analyzer import FlightAnalyzer

    log_data = LogParser(sys.argv[1]).parse()
    analyzer = FlightAnalyzer(log_data)
    phases, _ = analyzer.analyze()

    metrics_calc = PIDMetricsCalculator()

    print(f"\nPID Metrics Analysis:")
    for phase in phases:
        print(
            f"\nPhase: {phase.mode_name} ({phase.start_time:.1f}s - {phase.end_time:.1f}s)"
        )

        for axis_name, pid_data in [("Roll", log_data.roll), ("Pitch", log_data.pitch)]:
            if pid_data is None:
                continue

            phase_data = analyzer.extract_data_for_phase(pid_data, phase)
            if phase_data is None or phase_data.is_empty():
                continue

            tracking = metrics_calc.calculate_tracking(phase_data)
            oscillation = metrics_calc.analyze_dmod(phase_data)
            contributions = metrics_calc.analyze_contributions(phase_data)
            quality, score = metrics_calc.analyze_rate_tracking_quality(phase_data)

            print(f"\n  {axis_name}:")
            if tracking:
                print(f"    RMS Error: {tracking.rms_error:.3f} deg/s")
                print(f"    Max Error: {tracking.max_error:.2f} deg/s")
                print(f"    Correlation: {tracking.correlation:.3f}")
            if oscillation:
                print(f"    Dmod min: {oscillation.min_dmod:.3f}")
                print(f"    Oscillation: {oscillation.oscillation_severity}")
            if contributions:
                print(f"    FF contribution: {contributions.ff_pct:.1f}%")
            print(f"    Quality: {quality} ({score:.0f}/100)")
