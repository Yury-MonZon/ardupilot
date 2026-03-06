#!/usr/bin/env python3
"""
Advanced PID Analysis Methods

Mathematical approaches to find optimal PID tune:
1. System Identification - identify aircraft dynamics from flight data
2. Frequency Response Analysis - Bode plots from step responses
3. Optimal Control Theory - minimize tracking error
4. Iterative Learning - improve based on error patterns
"""

import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar, minimize
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SystemModel:
    """Identified aircraft dynamics model"""

    tau: float  # Time constant (seconds)
    gain: float  # DC gain (deg/s per deg deflection)
    delay: float  # Transport delay (seconds)
    damping: float  # Damping ratio
    natural_freq: float  # Natural frequency (rad/s)


@dataclass
class OptimalGains:
    """Calculated optimal PID gains"""

    FF: float
    P: float
    I: float
    D: float
    expected_rise_time: float
    expected_overshoot: float
    expected_settling_time: float


class AdvancedPIDAnalysis:
    """
    Advanced methods for PID analysis and optimization.
    """

    def __init__(self):
        pass

    def identify_system_dynamics(
        self,
        target: np.ndarray,
        actual: np.ndarray,
        time: np.ndarray,
        ff_output: np.ndarray,
    ) -> Optional[SystemModel]:
        """
        Identify aircraft dynamics from flight data using system identification.

        Uses the relationship between FF (feedforward) and actual rate response
        to estimate the aircraft's transfer function.

        The aircraft dynamics can be approximated as:
        G(s) = K * e^(-τs) / (τ_a * s + 1)

        Where:
        - K is the gain (rate per FF unit)
        - τ is the transport delay
        - τ_a is the aircraft time constant
        """
        # Only use data where there's actual control activity
        active_mask = np.abs(ff_output) > 0.1
        if np.sum(active_mask) < 100:
            return None

        target_active = target[active_mask]
        actual_active = actual[active_mask]
        ff_active = ff_output[active_mask]
        time_active = time[active_mask]

        # Calculate gain: ratio of actual rate to FF output
        # FF = k_ff * target, and actual ≈ gain * FF for steady state
        # So: gain = actual / FF (when tracking is good)

        # Use correlation method to estimate gain
        correlation = np.corrcoef(ff_active, actual_active)[0, 1]
        if correlation < 0.5:
            # Poor correlation, can't identify
            return None

        # Gain estimation using least squares
        # actual = gain * FF + noise
        gain = np.sum(ff_active * actual_active) / np.sum(ff_active * ff_active)

        # Time constant estimation from step response
        # Find step responses in the data
        dt = np.median(np.diff(time_active))

        # Calculate error to see how fast it converges
        error = np.abs(target_active - actual_active)

        # Estimate time constant from error decay
        # Use exponential fit to error peaks
        tau_estimate = self._estimate_time_constant(time_active, error)

        # Transport delay: time for response to start
        delay_estimate = self._estimate_delay(time_active, target_active, actual_active)

        return SystemModel(
            tau=tau_estimate,
            gain=gain,
            delay=delay_estimate,
            damping=0.7,  # Typical for well-tuned aircraft
            natural_freq=1.0 / tau_estimate if tau_estimate > 0 else 10.0,
        )

    def _estimate_time_constant(self, time: np.ndarray, error: np.ndarray) -> float:
        """Estimate system time constant from error decay"""
        # Find error peaks
        peaks = []
        for i in range(1, len(error) - 1):
            if error[i] > error[i - 1] and error[i] > error[i + 1] and error[i] > 0.5:
                peaks.append((time[i], error[i]))

        if len(peaks) < 3:
            return 0.3  # Default estimate

        # Fit exponential decay to peaks
        try:
            times = np.array([p[0] for p in peaks])
            errors = np.array([p[1] for p in peaks])

            # Normalize time
            t0 = times[0]
            times_norm = times - t0

            # Log of errors for linear fit
            log_errors = np.log(errors + 0.001)

            # Linear regression: log(e) = -t/tau + c
            A = np.vstack([times_norm, np.ones(len(times_norm))]).T
            result = np.linalg.lstsq(A, log_errors, rcond=None)
            slope = result[0][0]

            tau = -1.0 / slope if slope < 0 else 0.3
            tau = np.clip(tau, 0.1, 2.0)  # Reasonable bounds
            return tau
        except Exception:
            return 0.3

    def _estimate_delay(
        self, time: np.ndarray, target: np.ndarray, actual: np.ndarray
    ) -> float:
        """Estimate transport delay using cross-correlation"""
        try:
            # Compute cross-correlation
            correlation = np.correlate(
                (target - np.mean(target)) / (np.std(target) + 1e-6),
                (actual - np.mean(actual)) / (np.std(actual) + 1e-6),
                mode="full",
            )

            dt = np.median(np.diff(time))
            lags = np.arange(-len(target) + 1, len(target)) * dt

            # Find peak correlation
            mid = len(correlation) // 2
            # Only search in reasonable range (0 to 1 second delay)
            search_start = mid
            search_end = min(mid + int(1.0 / dt), len(correlation))

            if search_end <= search_start:
                return 0.05

            peak_idx = search_start + np.argmax(correlation[search_start:search_end])
            delay = lags[peak_idx]

            return max(0, delay)
        except Exception:
            return 0.05

    def calculate_optimal_gains(
        self, model: SystemModel, tau_target: float = 0.3, overshoot_target: float = 0.1
    ) -> OptimalGains:
        """
        Calculate optimal PID gains using control theory.

        For a first-order system G(s) = K / (τs + 1):

        1. FF = 1/K (feedforward for perfect tracking)
        2. P = τ / (K * τ_target) (for desired closed-loop time constant)
        3. I = P / (5 * τ_target) (for zero steady-state error)
        4. D ≈ 0.1 * P (typical ratio, adjust based on noise)

        The closed-loop transfer function with PID:
        T(s) = (K*(FF*s^2 + P*s + I)) / (τ*s^3 + (1 + K*D)*s^2 + K*P*s + K*I)

        Parameters:
            model: Identified system dynamics
            tau_target: Desired closed-loop time constant
            overshoot_target: Maximum acceptable overshoot (0-1)
        """
        K = model.gain if model.gain != 0 else 1.0
        tau = model.tau

        # Feedforward: for perfect tracking, FF should compensate for gain
        # FF output is multiplied by gain to get rate
        # So FF should be 1/K to get unity gain
        FF_optimal = 1.0 / K

        # P gain: for closed-loop time constant τ_target
        # Closed-loop pole at -1/τ_target requires P = τ/(K*τ_target)
        P_optimal = tau / (K * tau_target)

        # I gain: for zero steady-state error
        # Rule of thumb: I = P / (5*τ_target)
        I_optimal = P_optimal / (5 * tau_target)

        # D gain: for damping
        # Rule of thumb: D ≈ 0.1 * P, but depends on noise
        # Higher D if overshoot is a problem
        if overshoot_target < 0.05:
            D_optimal = 0.15 * P_optimal
        elif overshoot_target < 0.15:
            D_optimal = 0.1 * P_optimal
        else:
            D_optimal = 0.05 * P_optimal

        # Calculate expected performance
        # Rise time ≈ 2.2 * τ_target
        rise_time = 2.2 * tau_target

        # Overshoot depends on damping ratio
        # For PID, damping ≈ 0.7 is typical
        overshoot_pct = 100 * np.exp(-np.pi * 0.7 / np.sqrt(1 - 0.7**2))

        # Settling time ≈ 4.6 * τ_target
        settling_time = 4.6 * tau_target

        return OptimalGains(
            FF=FF_optimal,
            P=P_optimal,
            I=I_optimal,
            D=D_optimal,
            expected_rise_time=rise_time,
            expected_overshoot=overshoot_pct,
            expected_settling_time=settling_time,
        )

    def analyze_pd_balance(
        self, P: np.ndarray, D: np.ndarray, target: np.ndarray
    ) -> Dict:
        """
        Analyze P/D balance to understand tuning quality.

        For good tuning:
        - P should dominate during steady-state tracking
        - D should be active during transients
        - P/D ratio should be consistent

        From ArduPilot documentation:
        - P/D ratio of 3-5 is typical for fixed-wing
        - Higher P/D = more proportional response
        - Higher D/P = more damping
        """
        # Only analyze during active control
        active_mask = np.abs(target) > 3.0

        if not np.any(active_mask):
            return {"status": "no_active_control"}

        P_active = P[active_mask]
        D_active = D[active_mask]

        # P/D statistics
        P_mean = np.mean(np.abs(P_active))
        D_mean = np.mean(np.abs(D_active))
        P_max = np.max(np.abs(P_active))
        D_max = np.max(np.abs(D_active))

        pd_ratio_mean = P_mean / D_mean if D_mean > 1e-6 else float("inf")
        pd_ratio_peak = P_max / D_max if D_max > 1e-6 else float("inf")

        # Correlation between P and target (should be high for good tracking)
        P_target_corr = np.corrcoef(np.abs(P_active), np.abs(target[active_mask]))[0, 1]

        # D should correlate with target derivative, not target itself
        # Higher D during transients

        assessment = "good"
        if pd_ratio_mean < 1.5:
            assessment = "high_D"  # D dominates, might be too oscillatory
        elif pd_ratio_mean > 10:
            assessment = "low_D"  # P dominates, might overshoot
        elif P_target_corr < 0.3:
            assessment = "weak_P"  # P not tracking target well

        return {
            "status": "analyzed",
            "P_mean": P_mean,
            "D_mean": D_mean,
            "P_max": P_max,
            "D_max": D_max,
            "pd_ratio_mean": pd_ratio_mean,
            "pd_ratio_peak": pd_ratio_peak,
            "P_target_correlation": P_target_corr,
            "assessment": assessment,
        }

    def analyze_ff_efficiency(
        self,
        FF: np.ndarray,
        P: np.ndarray,
        D: np.ndarray,
        I: np.ndarray,
        actual: np.ndarray,
        target: np.ndarray,
    ) -> Dict:
        """
        Analyze FF efficiency.

        FF is the predictive term that should do most of the work
        when properly tuned. The ratio FF/(P+D+I) tells us how
        much work FF is doing vs. the feedback terms.

        Ideal: FF does 50-70% of the work
        - Higher FF% = better prediction, less feedback correction needed
        - Lower FF% = more feedback needed, might be sluggish
        """
        # Total control effort
        total = np.abs(FF) + np.abs(P) + np.abs(D) + np.abs(I)

        # Only analyze during active control
        active_mask = total > 0.01
        if not np.any(active_mask):
            return {"status": "no_control"}

        FF_contribution = (
            np.mean(np.abs(FF[active_mask])) / np.mean(total[active_mask]) * 100
        )
        P_contribution = (
            np.mean(np.abs(P[active_mask])) / np.mean(total[active_mask]) * 100
        )
        I_contribution = (
            np.mean(np.abs(I[active_mask])) / np.mean(total[active_mask]) * 100
        )
        D_contribution = (
            np.mean(np.abs(D[active_mask])) / np.mean(total[active_mask]) * 100
        )

        # FF efficiency assessment
        if FF_contribution > 60:
            ff_assessment = "excellent"  # FF doing most work
        elif FF_contribution > 45:
            ff_assessment = "good"
        elif FF_contribution > 30:
            ff_assessment = "fair"
        else:
            ff_assessment = "poor"  # Feedback doing too much work

        # Tracking quality
        tracking_error = np.mean(np.abs(actual - target))
        tracking_rms = np.sqrt(np.mean((actual - target) ** 2))

        return {
            "status": "analyzed",
            "FF_contribution": FF_contribution,
            "P_contribution": P_contribution,
            "I_contribution": I_contribution,
            "D_contribution": D_contribution,
            "ff_assessment": ff_assessment,
            "tracking_error_mean": tracking_error,
            "tracking_error_rms": tracking_rms,
        }

    def frequency_response(
        self, target: np.ndarray, actual: np.ndarray, time: np.ndarray
    ) -> Dict:
        """
        Compute frequency response to understand bandwidth.

        Uses FFT to find:
        - Bandwidth (frequency where response drops -3dB)
        - Phase margin (phase at crossover frequency)
        - Gain margin
        """
        dt = np.median(np.diff(time))
        n = len(time)

        # Compute FFTs
        target_fft = fft(target)
        actual_fft = fft(actual)

        # Frequency axis
        freqs = fftfreq(n, dt)

        # Transfer function (avoid division by zero)
        eps = 1e-10
        H = actual_fft / (target_fft + eps)

        # Only use positive frequencies
        pos_mask = freqs > 0

        freqs_pos = freqs[pos_mask]
        H_pos = H[pos_mask]

        # Magnitude in dB
        mag_db = 20 * np.log10(np.abs(H_pos) + 1e-10)

        # Phase in degrees
        phase = np.angle(H_pos) * 180 / np.pi

        # Find bandwidth (where magnitude drops -3dB)
        try:
            bandwidth_idx = np.where(mag_db < -3)[0]
            if len(bandwidth_idx) > 0:
                bandwidth = freqs_pos[bandwidth_idx[0]]
            else:
                bandwidth = freqs_pos[-1]
        except Exception:
            bandwidth = 0

        return {
            "status": "computed",
            "bandwidth_hz": bandwidth,
            "frequencies": freqs_pos[: min(100, len(freqs_pos))],
            "magnitude_db": mag_db[: min(100, len(mag_db))],
            "phase_deg": phase[: min(100, len(phase))],
        }

    def suggest_tuning_improvements(
        self,
        current_params: Dict,
        model: SystemModel,
        pd_analysis: Dict,
        ff_analysis: Dict,
    ) -> List[Dict]:
        """
        Generate specific tuning suggestions based on all analyses.
        """
        suggestions = []

        # FF suggestions
        if ff_analysis.get("ff_assessment") == "poor":
            suggestions.append(
                {
                    "priority": "HIGH",
                    "parameter": "FF",
                    "action": "increase",
                    "reason": f"FF contribution is {ff_analysis.get('FF_contribution', 0):.1f}% (should be >30%)",
                    "magnitude": 1.2,
                }
            )

        # P/D balance suggestions
        pd_status = pd_analysis.get("assessment", "good")
        if pd_status == "high_D":
            suggestions.append(
                {
                    "priority": "MEDIUM",
                    "parameter": "D",
                    "action": "decrease",
                    "reason": f"P/D ratio is {pd_analysis.get('pd_ratio_mean', 0):.1f} (D might be too high)",
                    "magnitude": 0.8,
                }
            )
        elif pd_status == "low_D":
            suggestions.append(
                {
                    "priority": "MEDIUM",
                    "parameter": "D",
                    "action": "increase",
                    "reason": f"P/D ratio is {pd_analysis.get('pd_ratio_mean', 0):.1f} (D might be too low)",
                    "magnitude": 1.3,
                }
            )

        # Optimal gains comparison
        optimal = self.calculate_optimal_gains(model)

        # Compare current vs optimal
        current_ff = current_params.get("FF", 0.15)
        current_p = current_params.get("P", 0.1)
        current_i = current_params.get("I", 0.1)
        current_d = current_params.get("D", 0.01)

        ff_ratio = current_ff / optimal.FF if optimal.FF > 0 else 1.0
        if ff_ratio < 0.7 or ff_ratio > 1.5:
            suggestions.append(
                {
                    "priority": "INFO",
                    "parameter": "FF",
                    "action": "adjust",
                    "reason": f"FF is {ff_ratio:.1f}x optimal value",
                    "current": current_ff,
                    "optimal": optimal.FF,
                }
            )

        return suggestions
