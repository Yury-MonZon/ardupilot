#!/usr/bin/env python3
"""
Visualizer - Generate plots for PID analysis
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
import os
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from log_parser import PIDData, ATRPData
from step_detector import Step, StepDetector, StepMetrics
from pid_metrics import TrackingMetrics, OscillationMetrics, PIDContributions


class Visualizer:
    """Generate analysis plots"""

    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Plot style
        plt.style.use("default")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3

    def plot_step_response(
        self,
        pid_data: PIDData,
        steps: List[Step],
        step_metrics_list: List[StepMetrics],
        axis_name: str,
        filename: str,
    ) -> str:
        """
        Plot step response analysis.

        Shows:
        - Target vs Actual rate for each detected step
        - Error over time
        - Step response metrics annotation
        """
        if pid_data.is_empty() or not steps:
            return ""

        n_steps = min(len(steps), 6)  # Max 6 steps per figure
        fig, axes = plt.subplots(n_steps, 1, figsize=(12, 3 * n_steps), squeeze=False)
        axes = axes.flatten()

        for i, (step, metrics) in enumerate(
            zip(steps[:n_steps], step_metrics_list[:n_steps])
        ):
            ax = axes[i]

            # Extract data around step
            margin = max(0.5, abs(step.step_size) * 0.05)
            start_time = step.start_time - margin
            end_time = step.end_time + margin * 3

            mask = (pid_data.time >= start_time) & (pid_data.time <= end_time)
            t = pid_data.time[mask]
            target = pid_data.target[mask]
            actual = pid_data.actual[mask]
            error = pid_data.error[mask]

            # Smooth actual for cleaner plot
            if len(actual) > 5:
                actual_smooth = self._smooth_array(actual, sigma=0.02)
            else:
                actual_smooth = actual

            # Plot
            ax.plot(t, target, "b-", label="Target", linewidth=1.5, alpha=0.8)
            ax.plot(t, actual_smooth, "g-", label="Actual", linewidth=1.5)
            ax.fill_between(
                t, target, actual_smooth, alpha=0.2, color="red", label="Error"
            )

            # Mark step region
            ax.axvline(
                step.start_time,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Step start",
            )
            ax.axvline(step.end_time, color="orange", linestyle="--", alpha=0.7)

            # Metrics annotation
            if metrics:
                metrics_text = (
                    f"Rise: {metrics.rise_time * 1000:.0f}ms\n"
                    f"Overshoot: {metrics.overshoot_pct:.1f}%\n"
                    f"Settling: {metrics.settling_time * 1000:.0f}ms"
                )
                ax.text(
                    0.02,
                    0.98,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Rate (deg/s)")
            ax.set_title(f"Step {i + 1}: {step.step_size:.1f} deg/s {step.direction}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def plot_pid_components(
        self,
        pid_data: PIDData,
        axis_name: str,
        filename: str,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Plot PID components over time.

        Shows:
        - P, I, D, FF components
        - Total output
        - Dmod indicator
        """
        if pid_data.is_empty():
            return ""

        # Apply time range if specified
        if time_range:
            mask = (pid_data.time >= time_range[0]) & (pid_data.time <= time_range[1])
            time = pid_data.time[mask]
            P = pid_data.P[mask]
            I = pid_data.I[mask]
            D = pid_data.D[mask]
            FF = pid_data.FF[mask]
            Dmod = pid_data.Dmod[mask]
        else:
            time = pid_data.time
            P = pid_data.P
            I = pid_data.I
            D = pid_data.D
            FF = pid_data.FF
            Dmod = pid_data.Dmod

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

        # PID Components
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time, P, "r-", label="P", linewidth=0.8, alpha=0.8)
        ax1.plot(time, I, "g-", label="I", linewidth=0.8, alpha=0.8)
        ax1.plot(time, D, "b-", label="D", linewidth=0.8, alpha=0.8)
        ax1.plot(time, FF, "m-", label="FF", linewidth=0.8, alpha=0.8)

        total = P + I + D + FF
        ax1.plot(time, total, "k-", label="Total", linewidth=1.0, alpha=0.7)

        ax1.set_ylabel("Component Value (deg)")
        ax1.set_title(f"{axis_name} Rate PID Components")
        ax1.legend(loc="upper right", ncol=5)
        ax1.grid(True, alpha=0.3)

        # Dmod
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(time, Dmod, "c-", linewidth=0.8)
        ax2.axhline(
            1.0, color="red", linestyle="--", alpha=0.7, label="Oscillation threshold"
        )
        ax2.axhline(
            0.9, color="orange", linestyle="--", alpha=0.7, label="Minor oscillation"
        )
        ax2.fill_between(time, Dmod, 1.0, where=Dmod < 1.0, alpha=0.3, color="red")

        ax2.set_ylabel("Dmod")
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Target vs Actual
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if time_range:
            mask = (pid_data.time >= time_range[0]) & (pid_data.time <= time_range[1])
            ax3.plot(
                pid_data.time[mask],
                pid_data.target[mask],
                "b-",
                label="Target",
                linewidth=0.8,
            )
            ax3.plot(
                pid_data.time[mask],
                pid_data.actual[mask],
                "g-",
                label="Actual",
                linewidth=0.8,
            )
        else:
            ax3.plot(
                pid_data.time, pid_data.target, "b-", label="Target", linewidth=0.8
            )
            ax3.plot(
                pid_data.time, pid_data.actual, "g-", label="Actual", linewidth=0.8
            )

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Rate (deg/s)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def plot_autotune_progression(self, atrp_data: ATRPData, filename: str) -> str:
        """
        Plot autotune gain progression from ATRP messages.

        Shows how FF, P, D evolved during autotune.
        """
        if atrp_data is None or len(atrp_data.time) == 0:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for axis_idx, (axis_name, axis_mask) in enumerate(
            [("Roll", atrp_data.axis == 0), ("Pitch", atrp_data.axis == 1)]
        ):
            if not np.any(axis_mask):
                continue

            time = atrp_data.time[axis_mask]
            FF = atrp_data.FF[axis_mask]
            P = atrp_data.P[axis_mask]
            D = atrp_data.D[axis_mask]
            I = atrp_data.I[axis_mask]

            ax = axes[axis_idx, 0]
            ax.plot(time, FF, "m-", label="FF", linewidth=1.5)
            ax.plot(time, P, "r-", label="P", linewidth=1.5)
            ax.plot(time, I, "g-", label="I", linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Gain")
            ax.set_title(f"{axis_name} Autotune - Gains")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[axis_idx, 1]
            ax.plot(time, D, "b-", label="D", linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("D Gain")
            ax.set_title(f"{axis_name} Autotune - D Gain")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def plot_error_distribution(
        self, pid_data: PIDData, axis_name: str, filename: str
    ) -> str:
        """
        Plot error distribution histogram.
        """
        if pid_data.is_empty():
            return ""

        error = pid_data.error

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax = axes[0]
        bins = np.linspace(-10, 10, 51)
        ax.hist(
            error,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # Fit and plot Gaussian
        mean = np.mean(error)
        std = np.std(error)
        x = np.linspace(-10, 10, 100)
        ax.plot(
            x,
            1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2)),
            "r-",
            linewidth=2,
            label=f"Gaussian fit\nμ={mean:.3f}\nσ={std:.3f}",
        )

        ax.set_xlabel("Error (deg/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{axis_name} Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Time series of error
        ax = axes[1]
        ax.plot(pid_data.time, error, "b-", linewidth=0.5, alpha=0.7)
        ax.axhline(mean, color="red", linestyle="--", label=f"Mean: {mean:.3f}")
        ax.axhline(
            mean + std, color="orange", linestyle=":", label=f"+/- 1σ: {std:.3f}"
        )
        ax.axhline(mean - std, color="orange", linestyle=":")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (deg/s)")
        ax.set_title(f"{axis_name} Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def plot_tracking_scatter(
        self, pid_data: PIDData, axis_name: str, filename: str
    ) -> str:
        """
        Plot target vs actual scatter plot.
        """
        if pid_data.is_empty():
            return ""

        fig, ax = plt.subplots(figsize=(8, 8))

        # Subsample for large datasets
        if len(pid_data.target) > 50000:
            idx = np.linspace(0, len(pid_data.target) - 1, 50000, dtype=int)
            target = pid_data.target[idx]
            actual = pid_data.actual[idx]
        else:
            target = pid_data.target
            actual = pid_data.actual

        # Scatter plot with density coloring
        hb = ax.hexbin(target, actual, gridsize=50, cmap="YlOrRd", mincnt=1)
        cb = plt.colorbar(hb, ax=ax, label="Count")

        # Ideal line
        lim = max(np.max(np.abs(target)), np.max(np.abs(actual))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "b--", linewidth=2, label="Ideal", alpha=0.7)

        ax.set_xlabel("Target Rate (deg/s)")
        ax.set_ylabel("Actual Rate (deg/s)")
        ax.set_title(f"{axis_name} Tracking Accuracy")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def _smooth_array(self, data: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        """Apply Gaussian smoothing to array"""
        if len(data) < 5:
            return data

        # Estimate sample rate
        sigma_samples = max(int(len(data) * sigma * 10), 3)
        if sigma_samples % 2 == 0:
            sigma_samples += 1

        kernel = np.exp(-(np.linspace(-3, 3, sigma_samples) ** 2) / 2)
        kernel = kernel / kernel.sum()

        return np.convolve(data, kernel, mode="same")

    def plot_power_curve(
        self,
        bin_speeds_kmh: np.ndarray,
        bin_efficiencies: np.ndarray,
        bin_powers: np.ndarray,
        raw_speeds_kmh: np.ndarray,
        raw_efficiencies: np.ndarray,
        raw_powers: np.ndarray,
        best_range_speed: float,
        best_endurance_speed: float,
        filename: str,
        use_airspeed: bool = False,
    ) -> str:
        """
        Plot power curve: efficiency and power vs speed.
        Shows raw data as scatter, binned averages, and fitted curves.
        
        Args:
            use_airspeed: If True, indicates airspeed (wind-agnostic) was used.
                         If False, GPS groundspeed was used.
        """
        if len(bin_speeds_kmh) == 0 or len(raw_speeds_kmh) == 0:
            return ""

        # Filter out NaN values from binned data
        valid = ~np.isnan(bin_efficiencies) & ~np.isnan(bin_powers)
        if np.sum(valid) < 5:
            return ""

        bin_speeds = bin_speeds_kmh[valid]
        bin_eff = bin_efficiencies[valid]
        bin_pwr = bin_powers[valid]

        # Fit curves to raw data
        try:
            eff_coeffs = np.polyfit(raw_speeds_kmh, raw_efficiencies, deg=2)
            eff_fit = np.poly1d(eff_coeffs)
            power_coeffs = np.polyfit(raw_speeds_kmh, raw_powers, deg=3)
            power_fit = np.poly1d(power_coeffs)

            speed_range = np.linspace(raw_speeds_kmh.min(), raw_speeds_kmh.max(), 200)
            eff_curve = eff_fit(speed_range)
            pwr_curve = power_fit(speed_range)
            has_fit = True
        except Exception:
            has_fit = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Speed source indicator
        speed_source = "Airspeed (wind-agnostic)" if use_airspeed else "GPS Groundspeed"

        # Efficiency plot
        # Raw data scatter (light)
        ax1.scatter(
            raw_speeds_kmh,
            raw_efficiencies,
            alpha=0.15,
            s=2,
            c="blue",
            label=f"Raw data (n={len(raw_speeds_kmh):,})",
        )
        # Binned averages
        ax1.plot(
            bin_speeds,
            bin_eff,
            "b-o",
            linewidth=2,
            markersize=6,
            label="Binned average",
        )
        # Fitted curve
        if has_fit:
            ax1.plot(speed_range, eff_curve, "g-", linewidth=2.5, label="Quadratic fit")
        # Best range speed marker
        ax1.axvline(
            best_range_speed,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best Range: {best_range_speed:.1f} km/h",
        )
        ax1.set_ylabel("Efficiency (Wh/km)")
        ax1.set_title(f"Power Curve Analysis - {speed_source}")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Power plot
        # Raw data scatter (light)
        ax2.scatter(
            raw_speeds_kmh,
            raw_powers,
            alpha=0.15,
            s=2,
            c="red",
            label=f"Raw data (n={len(raw_speeds_kmh):,})",
        )
        # Binned averages
        ax2.plot(
            bin_speeds,
            bin_pwr,
            "r-o",
            linewidth=2,
            markersize=6,
            label="Binned average",
        )
        # Fitted curve
        if has_fit:
            ax2.plot(speed_range, pwr_curve, "orange", linewidth=2.5, label="Cubic fit")
        # Best endurance speed marker
        ax2.axvline(
            best_endurance_speed,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Best Endurance: {best_endurance_speed:.1f} km/h",
        )
        ax2.set_xlabel("Speed (km/h)")
        ax2.set_ylabel("Power (W)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def generate_all_plots(
        self,
        roll_data: PIDData,
        pitch_data: PIDData,
        atrp_data: Optional[ATRPData],
        roll_steps: List[Step],
        pitch_steps: List[Step],
        roll_metrics: Optional[StepMetrics],
        pitch_metrics: Optional[StepMetrics],
        power_curve_data: Optional[Tuple] = None,
        use_airspeed: bool = False,
    ) -> Dict[str, str]:
        """Generate all analysis plots
        
        Args:
            use_airspeed: If True, indicates airspeed (wind-agnostic) was used for power curve.
        """
        plots = {}

        # Step response plots
        if roll_steps:
            roll_metrics_list = (
                [roll_metrics] if roll_metrics else [None] * len(roll_steps)
            )
            path = self.plot_step_response(
                roll_data,
                roll_steps,
                roll_metrics_list,
                "Roll",
                "roll_step_response.png",
            )
            if path:
                plots["roll_step_response"] = path

        if pitch_steps:
            pitch_metrics_list = (
                [pitch_metrics] if pitch_metrics else [None] * len(pitch_steps)
            )
            path = self.plot_step_response(
                pitch_data,
                pitch_steps,
                pitch_metrics_list,
                "Pitch",
                "pitch_step_response.png",
            )
            if path:
                plots["pitch_step_response"] = path

        # PID components
        if roll_data and not roll_data.is_empty():
            path = self.plot_pid_components(
                roll_data, "Roll", "roll_pid_components.png"
            )
            if path:
                plots["roll_pid_components"] = path

        if pitch_data and not pitch_data.is_empty():
            path = self.plot_pid_components(
                pitch_data, "Pitch", "pitch_pid_components.png"
            )
            if path:
                plots["pitch_pid_components"] = path

        # Autotune progression
        if atrp_data and len(atrp_data.time) > 0:
            path = self.plot_autotune_progression(atrp_data, "autotune_progression.png")
            if path:
                plots["autotune_progression"] = path

        # Error distributions
        if roll_data and not roll_data.is_empty():
            path = self.plot_error_distribution(
                roll_data, "Roll", "roll_error_distribution.png"
            )
            if path:
                plots["roll_error_distribution"] = path

        if pitch_data and not pitch_data.is_empty():
            path = self.plot_error_distribution(
                pitch_data, "Pitch", "pitch_error_distribution.png"
            )
            if path:
                plots["pitch_error_distribution"] = path

        # Tracking scatter
        if roll_data and not roll_data.is_empty():
            path = self.plot_tracking_scatter(
                roll_data, "Roll", "roll_tracking_scatter.png"
            )
            if path:
                plots["roll_tracking_scatter"] = path

        if pitch_data and not pitch_data.is_empty():
            path = self.plot_tracking_scatter(
                pitch_data, "Pitch", "pitch_tracking_scatter.png"
            )
            if path:
                plots["pitch_tracking_scatter"] = path

        # Power curve
        if power_curve_data:
            (
                bin_speeds,
                bin_eff,
                bin_pwr,
                raw_speeds,
                raw_eff,
                raw_pwr,
                best_range,
                best_endurance,
            ) = power_curve_data
            path = self.plot_power_curve(
                bin_speeds,
                bin_eff,
                bin_pwr,
                raw_speeds,
                raw_eff,
                raw_pwr,
                best_range,
                best_endurance,
                "power_curve.png",
                use_airspeed=use_airspeed,
            )
            if path:
                plots["power_curve"] = path

        return plots


if __name__ == "__main__":
    print("Visualizer module - use via nv_pid_analyzer.py")
