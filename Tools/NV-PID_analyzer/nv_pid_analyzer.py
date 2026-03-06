#!/usr/bin/env python3
"""
NV-PID Analyzer - ArduPlane PID/FF Parameter Analysis Tool

Analyzes binary logs to evaluate PIDFF tuning and provide suggestions.
Supports:
- Roll and Pitch rate controller analysis
- Step response metrics (rise time, overshoot, settling time)
- Oscillation detection via Dmod
- Tuning suggestions based on ArduPilot autotune logic
- Autotune session analysis

Usage:
    python nv_pid_analyzer.py <logfile.bin> [options]

Examples:
    python nv_pid_analyzer.py log1.bin
    python nv_pid_analyzer.py log1.bin --axis roll
    python nv_pid_analyzer.py log1.bin --output ./my_plots
"""

import argparse
import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log_parser import LogParser, LogData, PIDData
from flight_analyzer import FlightAnalyzer, AnalysisPhase, AutotunePhysics
from step_detector import StepDetector, Step, StepMetrics
from pid_metrics import (
    PIDMetricsCalculator,
    TrackingMetrics,
    OscillationMetrics,
    PIDContributions,
)
from tuning_suggester import TuningSuggester, Suggestion
from visualizer import Visualizer
from power_curve_analyzer import PowerCurveAnalyzer


def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header"""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_section(title: str, char: str = "-", width: int = 80):
    """Print a formatted section header"""
    print()
    print(f" {title}")
    print(char * (width - 1))


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def analyze_axis(
    log_data: LogData,
    axis: str,
    phases: List[AnalysisPhase],
    metrics_calc: PIDMetricsCalculator,
    flight_analyzer: FlightAnalyzer,
) -> Dict:
    """
    Analyze one axis (roll or pitch) across all analysis phases.

    Returns dict with:
    - combined_data: PIDData merged from all phases
    - tracking: TrackingMetrics
    - oscillation: OscillationMetrics
    - contributions: PIDContributions
    - steps: List of detected steps
    - step_metrics: Average StepMetrics
    """
    pid_data = log_data.roll if axis.lower() == "roll" else log_data.pitch

    if pid_data is None or pid_data.is_empty():
        return None

    # Combine data from all analysis phases
    combined = PIDData()

    for phase in phases:
        phase_data = flight_analyzer.extract_data_for_phase(pid_data, phase)
        if phase_data and not phase_data.is_empty():
            combined.time = (
                np.concatenate([combined.time, phase_data.time])
                if len(combined.time) > 0
                else phase_data.time.copy()
            )
            combined.target = (
                np.concatenate([combined.target, phase_data.target])
                if len(combined.target) > 0
                else phase_data.target.copy()
            )
            combined.actual = (
                np.concatenate([combined.actual, phase_data.actual])
                if len(combined.actual) > 0
                else phase_data.actual.copy()
            )
            combined.error = (
                np.concatenate([combined.error, phase_data.error])
                if len(combined.error) > 0
                else phase_data.error.copy()
            )
            combined.P = (
                np.concatenate([combined.P, phase_data.P])
                if len(combined.P) > 0
                else phase_data.P.copy()
            )
            combined.I = (
                np.concatenate([combined.I, phase_data.I])
                if len(combined.I) > 0
                else phase_data.I.copy()
            )
            combined.D = (
                np.concatenate([combined.D, phase_data.D])
                if len(combined.D) > 0
                else phase_data.D.copy()
            )
            combined.FF = (
                np.concatenate([combined.FF, phase_data.FF])
                if len(combined.FF) > 0
                else phase_data.FF.copy()
            )
            combined.DFF = (
                np.concatenate([combined.DFF, phase_data.DFF])
                if len(combined.DFF) > 0
                else phase_data.DFF.copy()
            )
            combined.Dmod = (
                np.concatenate([combined.Dmod, phase_data.Dmod])
                if len(combined.Dmod) > 0
                else phase_data.Dmod.copy()
            )
            combined.SRate = (
                np.concatenate([combined.SRate, phase_data.SRate])
                if len(combined.SRate) > 0
                else phase_data.SRate.copy()
            )
            combined.Flags = (
                np.concatenate([combined.Flags, phase_data.Flags])
                if len(combined.Flags) > 0
                else phase_data.Flags.copy()
            )

    if combined.is_empty():
        return None

    # Recalculate sample rate
    if len(combined.time) > 1:
        dt = np.diff(combined.time)
        combined.sample_rate = 1.0 / np.median(dt) if np.median(dt) > 0 else 0.0

    # Calculate metrics
    tracking = metrics_calc.calculate_tracking(combined)
    oscillation = metrics_calc.analyze_dmod(combined)
    contributions = metrics_calc.analyze_contributions(combined)

    # Detect steps
    detector = StepDetector(combined)
    steps = detector.detect_steps(min_step_deg_s=10.0)
    step_metrics = detector.calculate_average_metrics(steps) if steps else None

    # Assess step quality
    step_quality_score, step_quality_msg = (
        detector.assess_step_quality(steps) if steps else (0, "No steps detected")
    )

    return {
        "combined_data": combined,
        "tracking": tracking,
        "oscillation": oscillation,
        "contributions": contributions,
        "steps": steps,
        "step_metrics": step_metrics,
        "step_quality_score": step_quality_score,
        "step_quality_msg": step_quality_msg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NV-PID Analyzer - ArduPlane PID/FF Parameter Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s log1.bin
  %(prog)s log1.bin --axis roll
  %(prog)s log1.bin --output ./my_plots
        """,
    )

    parser.add_argument("logfile", help="Path to ArduPilot .bin log file")
    parser.add_argument(
        "--axis",
        choices=["roll", "pitch", "both"],
        default="both",
        help="Axis to analyze (default: both)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./plots",
        help="Output directory for plots (default: ./plots)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Parse log
    print_header("NV-PID ANALYZER")
    print(f"\nLog File: {args.logfile}")

    try:
        log_parser = LogParser(args.logfile)
        log_data = log_parser.parse()
    except FileNotFoundError:
        print(f"Error: Log file not found: {args.logfile}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing log: {e}")
        sys.exit(1)

    # Analyze flight phases
    flight_analyzer = FlightAnalyzer(log_data)

    # Get autotune sessions and prompt user if multiple
    sessions = flight_analyzer.get_autotune_sessions()
    session_idx = None

    if sessions:
        print(f"\nAutotune sessions found: {len(sessions)}")
        for s in sessions:
            status = []
            if s.has_roll:
                status.append("roll OK")
            if s.has_pitch:
                status.append("pitch OK")
            status_str = ", ".join(status) if status else "incomplete"
            print(
                f"  Session {s.index}: {s.start_time:.0f}s - {s.end_time:.0f}s ({s.duration_s:.0f}s) [{status_str}]"
            )

        # Prompt user for session selection
        try:
            choice = input(
                f"\nSelect session (1-{len(sessions)}, or Enter for 1): "
            ).strip()
            if choice == "":
                session_idx = 1
            else:
                session_idx = int(choice)
        except (EOFError, ValueError):
            session_idx = 1
            print(f"Using session 1")

    phases, autotune_physics = flight_analyzer.analyze(session_idx)

    # Get autotune level
    autotune_level = int(log_data.params.get("AUTOTUNE_LEVEL", 7))

    # Print parameters
    print_header("PARAMETERS")
    print(f"\nAUTOTUNE_LEVEL: {autotune_level}")
    tau, rmax, _, _ = flight_analyzer.get_targets_for_level(autotune_level)
    print(f"  (tau={tau:.2f}s, rmax={rmax} deg/s)")

    for axis, prefix in [("Roll", "RLL"), ("Pitch", "PTCH")]:
        p = log_data.params.get(f"{prefix}_RATE_P", 0)
        i = log_data.params.get(f"{prefix}_RATE_I", 0)
        d = log_data.params.get(f"{prefix}_RATE_D", 0)
        ff = log_data.params.get(f"{prefix}_RATE_FF", 0)
        imax = log_data.params.get(f"{prefix}_RATE_IMAX", 0)

        print(f"\n{axis} Rate PID:")
        print(f"  FF={ff:.4f} P={p:.4f} I={i:.4f} D={d:.4f} IMAX={imax:.3f}")

    # Print flight analysis
    print_header("FLIGHT ANALYSIS")

    if log_data.autotune_events:
        print("\nAutotune sessions detected:")
        for event in log_data.autotune_events:
            print(f"  [{event.time_s:.1f}s] {event.event_type}: {event.message}")

    print(f"\nAnalysis phases ({len(phases)} total):")
    for phase in phases:
        print(
            f" [{phase.start_time:.1f}s - {phase.end_time:.1f}s] {phase.mode_name} "
            f"({format_duration(phase.duration_s)}) {phase.phase_type}"
        )

    # Power Curve Analysis
    power_result = None
    if log_data.battery and log_data.gps:
        print_header("POWER CURVE ANALYSIS")

        airspeed_min = log_data.params.get("AIRSPEED_MIN", 10.0)
        power_analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=0.5)

        # Use airspeed if available (wind-agnostic), otherwise GPS groundspeed
        power_data = power_analyzer.extract_power_data(
            log_data.battery, log_data.gps, log_data.baro, log_data.airspeed
        )

        # Debug: show data counts
        speed_source = "Airspeed (wind-agnostic)" if log_data.airspeed else "GPS Groundspeed"
        print(f"Speed Source: {speed_source}")
        print(f"  Battery samples: {len(log_data.battery)}")
        print(f"  GPS samples: {len(log_data.gps)}")
        print(f"  Airspeed samples: {len(log_data.airspeed)}")
        print(f"  Power data points extracted: {len(power_data)}")

        power_result = power_analyzer.analyze(power_data)

        if power_result:
            print(
                f"\nBest Range Speed: {power_result.best_range_speed_kmh:.1f} km/h "
                f"({power_result.best_range_speed_mps:.1f} m/s) "
                f"(data max: {power_result.speed_range_mps[1]:.1f} m/s)"
            )
            print(
                f"  Efficiency: {power_result.best_range_efficiency_wh_per_km:.3f} Wh/km (maximum distance per battery)"
            )
            print(f"  Power: {power_result.best_range_power_w:.0f} W")
            print(
                f"\nBest Endurance Speed: {power_result.best_endurance_speed_kmh:.1f} km/h "
                f"({power_result.best_endurance_speed_mps:.1f} m/s) "
                f"(data min: {power_result.speed_range_mps[0]:.1f} m/s)"
            )
            print(
                f"  Power: {power_result.best_endurance_power_w:.0f} W (maximum flight time)"
            )
            print(
                f"  Efficiency: {power_result.best_endurance_efficiency_wh_per_km:.3f} Wh/km"
            )
            print(f"\nData quality:")
            print(
                f"  Valid samples: {power_result.valid_data_points:,} / {power_result.total_data_points:,}"
            )
            print(
                f"  Speed range: {power_result.speed_range_kmh[0]:.1f} - {power_result.speed_range_kmh[1]:.1f} km/h"
            )
            print(
                f"  Efficiency range: {power_result.efficiency_range_wh_per_km[0]:.3f} - "
                f"{power_result.efficiency_range_wh_per_km[1]:.3f} Wh/km"
            )
        else:
            print("\nInsufficient data for power curve analysis")
            print("  (need stable cruise conditions with battery and GPS data)")
            print("  Tip: Ensure BAT (battery) and GPS messages are logged")
            print("  Check ARSP logging if airspeed sensor is installed")
    else:
        print_header("POWER CURVE ANALYSIS")
        if not log_data.battery:
            print("\nNo battery (BAT) data found in log")
            print("  Power curve analysis requires battery voltage/current logging")
        if not log_data.gps:
            print("\nNo GPS data found in log")
            print("  Power curve analysis requires GPS speed logging")
        print("  Tip: Add these to your log_bitmask parameters:")
        print("    - BAT: Log battery voltage and current")
        print("    - GPS: Log GPS speed")
        print("    - ARSP: Log airspeed (for wind-agnostic analysis)")

    # Analyze axes
    metrics_calc = PIDMetricsCalculator()

    axes_to_analyze = []
    if args.axis in ["roll", "both"]:
        axes_to_analyze.append("Roll")
    if args.axis in ["pitch", "both"]:
        axes_to_analyze.append("Pitch")

    results = {}

    for axis in axes_to_analyze:
        print_header(f"{axis.upper()} RATE ANALYSIS")

        axis_results = analyze_axis(
            log_data, axis.lower(), phases, metrics_calc, flight_analyzer
        )

        if axis_results is None:
            print(f"\nNo data available for {axis}")
            continue

        results[axis] = axis_results

        tracking = axis_results["tracking"]
        oscillation = axis_results["oscillation"]
        contributions = axis_results["contributions"]
        steps = axis_results["steps"]
        step_metrics = axis_results["step_metrics"]
        combined = axis_results["combined_data"]

        # Print tracking performance
        print(
            f"\nAnalyzed {len(combined.time)} samples at {combined.sample_rate:.1f} Hz"
        )

        if tracking:
            print("\nTracking Performance:")
            print(f"  RMS Error: {tracking.rms_error:.3f} deg/s", end="")
            print(" ✓" if tracking.rms_error < 1.0 else "")
            print(f"  Mean Error: {tracking.mean_error:.3f} deg/s (bias)")
            print(f"  Max Error: {tracking.max_error:.2f} deg/s")
            print(f"  Correlation: {tracking.correlation:.4f}")

        # Print oscillation check
        if oscillation:
            print("\nOscillation Check:")
            print(f"  Dmod Min: {oscillation.min_dmod:.3f}", end="")
            if oscillation.oscillation_detected:
                print(f" ⚠ ({oscillation.oscillation_severity})")
            else:
                print(" ✓ (no oscillation)")
        print(f" Dmod Mean: {oscillation.mean_dmod:.3f}")

        # Use autotune data if available - it has clean step inputs
        has_good_autotune_data = False
        if log_data.atrp and len(log_data.atrp.time) > 0:
            from autotune_analyzer import AutotuneAnalyzer, PIDGains

            at_analyzer = AutotuneAnalyzer(log_data.atrp)
            at_steps = at_analyzer.extract_steps()

            axis_num = 0 if axis.lower() == "roll" else 1
            good_at_steps = at_analyzer.get_good_steps(axis_num)

            if good_at_steps:
                has_good_autotune_data = True
                print(f"\\nAutotune Step Analysis (clean step inputs):")
                print(f" Good autotune steps: {len(good_at_steps)}")

                best_gains, gain_scores = at_analyzer.find_optimal_gains(axis_num)
                if best_gains and best_gains in gain_scores:
                    s = gain_scores[best_gains]
                    print(f" Best autotune gains:")
                    print(
                        f"  FF={best_gains.ff:.4f} P={best_gains.p:.4f} I={best_gains.i:.4f} D={best_gains.d:.4f}"
                    )
                    print(f"  Avg overshoot: {s['avg_overshoot']:.1f}%")
                    print(f"  Avg rise time: {s['avg_rise_time'] * 1000:.0f}ms")
                    print(f"  Quality score: {s['avg_quality']:.0f}/100")

                    # Compare with current parameters
                    prefix = "RLL" if axis_num == 0 else "PTCH"
                    current_ff = log_data.params.get(f"{prefix}_RATE_FF", 0)
                    current_p = log_data.params.get(f"{prefix}_RATE_P", 0)
                    current_d = log_data.params.get(f"{prefix}_RATE_D", 0)

                    if (
                        abs(current_ff - best_gains.ff) > 0.001
                        or abs(current_p - best_gains.p) > 0.001
                    ):
                        print(f"\\n  Current gains differ from best autotune gains:")
                        print(
                            f"  Current: FF={current_ff:.4f} P={current_p:.4f} D={current_d:.4f}"
                        )
            else:
                print("\\n No good autotune steps found for this axis")

        # Only show flight-data step analysis if no good autotune data
        if not has_good_autotune_data and steps and step_metrics:
            from step_detector import StepDetector

            temp_detector = StepDetector(combined)
            step_quality, step_quality_msg = temp_detector.assess_step_quality(steps)

            print(f"\\nFlight Data Step Analysis:")
            print(f" Steps detected: {len(steps)}")
            print(f" Step quality: {step_quality_msg}")

            if step_quality >= 50:
                print(f"\\n ⚠ Note: Flight data may not have clean step inputs.")
                print(f" Rise time and overshoot metrics may be unreliable.")
                print(f" Focus on tracking RMS error and Dmod instead.")
            else:
                print(
                    f"\\n ⚠ Step metrics unreliable - focus on tracking quality instead."
                )

        # Print PID contributions
        if contributions:
            print("\nPID Component Contribution:")
            print(f" FF: {contributions.ff_pct:.1f}%", end="")
            print(" (feedforward dominant)" if contributions.ff_pct > 50 else "")
            print(f" P: {contributions.p_pct:.1f}%")
            print(f" I: {contributions.i_pct:.1f}%")
            print(f" D: {contributions.d_pct:.1f}%")

            # P/D ratio analysis
            pd_ratio = (
                contributions.p_pct / contributions.d_pct
                if contributions.d_pct > 0
                else float("inf")
            )
            print(f"\nP/D Ratio: {pd_ratio:.2f}:1")
            if pd_ratio > 8:
                print("  → High P/D: P dominates, consider more D for damping")
            elif pd_ratio < 1.5:
                print("  → Low P/D: D high relative to P, may cause over-damping")
            elif 3 <= pd_ratio <= 5:
                print("  → Optimal range (3-5): Balanced P and D contributions")
            else:
                print("  → Acceptable: Typical fixed-wing range is 3-5")

        # Speed scaler analysis
        if log_data.speed_scaler and len(phases) > 0:
            phase_start = phases[0].start_time if phases else 0
            phase_end = phases[-1].end_time if phases else log_data.duration_s
            scaler_analysis = flight_analyzer.analyze_scaler_effects(
                combined, log_data.speed_scaler, phase_start, phase_end
            )
            if scaler_analysis and scaler_analysis.get("scaler_available"):
                print("\nSpeed Scaler Analysis:")
                print(
                    f" Scaler range: {scaler_analysis['scaler_range'][0]:.2f} - {scaler_analysis['scaler_range'][1]:.2f}"
                )
                print(f" Mean scaler: {scaler_analysis['scaler_mean']:.2f}")
                if "low_speed" in scaler_analysis:
                    ls = scaler_analysis["low_speed"]
                    print(
                        f" Low speed (scaler>1.2): mean error={ls['mean_error']:.2f} deg/s ({ls['sample_count']} samples)"
                    )
                if "cruise" in scaler_analysis:
                    cs = scaler_analysis["cruise"]
                    print(
                        f" Cruise (scaler 0.9-1.1): mean error={cs['mean_error']:.2f} deg/s ({cs['sample_count']} samples)"
                    )
                if "high_speed" in scaler_analysis:
                    hs = scaler_analysis["high_speed"]
                    print(
                        f" High speed (scaler<0.8): mean error={hs['mean_error']:.2f} deg/s ({hs['sample_count']} samples)"
                    )
                print(f" Analysis: {scaler_analysis.get('analysis', 'N/A')}")

        # Quality assessment
        quality, score = metrics_calc.analyze_rate_tracking_quality(combined)
        print(f"\nOverall Quality: {quality.upper()} ({score:.0f}/100)")

    # Generate suggestions
    print_header("TUNING SUGGESTIONS")

    suggester = TuningSuggester(log_data.params, autotune_level, autotune_physics)

    for axis in axes_to_analyze:
        if axis not in results:
            continue

        print_section(f"{axis} SUGGESTIONS")

        axis_results = results[axis]

        # Don't use step metrics if quality is poor and no autotune data
        # Need very high quality (>=80) for flight data, since it rarely has clean step inputs
        step_quality = axis_results.get("step_quality_score", 100)
        has_autotune = log_data.atrp is not None and len(log_data.atrp.time) > 0
        use_step_metrics = (
            axis_results["step_metrics"]
            if (has_autotune or step_quality >= 80)
            else None
        )

        suggestions = suggester.suggest(
            axis=axis,
            tracking=axis_results["tracking"],
            oscillation=axis_results["oscillation"],
            step_metrics=use_step_metrics,
            ff_contribution=axis_results["contributions"].ff_pct
            if axis_results["contributions"]
            else 0,
            step_quality_score=step_quality,
        )

        print(suggester.format_suggestions(suggestions))

    # Summary
    print_header("SUMMARY")

    critical_count = 0
    for axis_name, axis_results in results.items():
        if axis_results is None:
            continue
        suggestions = suggester.suggest(
            axis_name.lower(),
            axis_results["tracking"],
            axis_results["oscillation"],
            axis_results["step_metrics"],
            axis_results["contributions"].ff_pct
            if axis_results["contributions"]
            else 0,
        )
        critical_count += sum(1 for s in suggestions if s.priority == "CRITICAL")

    if critical_count == 0:
        print("\n✓ No critical issues detected")
    else:
        print(f"\n⚠ {critical_count} critical issue(s) require attention")

    # Generate plots
    if not args.no_plots and results:
        print(f"\nGenerating plots...")
        visualizer = Visualizer(args.output)

        roll_data = results.get("Roll", {}).get("combined_data")
        pitch_data = results.get("Pitch", {}).get("combined_data")
        roll_steps = results.get("Roll", {}).get("steps", [])
        pitch_steps = results.get("Pitch", {}).get("steps", [])
        roll_metrics = results.get("Roll", {}).get("step_metrics")
        pitch_metrics = results.get("Pitch", {}).get("step_metrics")

        # Power curve data for plotting
        power_curve_plot_data = None
        if log_data.battery and log_data.gps and power_result:
            airspeed_min = log_data.params.get("AIRSPEED_MIN", 10.0)
            power_analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=0.5)
            # Use airspeed if available (wind-agnostic), otherwise GPS groundspeed
            power_data = power_analyzer.extract_power_data(
                log_data.battery, log_data.gps, log_data.baro, log_data.airspeed
            )
            (bin_speeds, bin_eff, bin_pwr, raw_speeds, raw_eff, raw_pwr) = (
                power_analyzer.get_binned_data(power_data)
            )
            if len(bin_speeds) > 0:
                power_curve_plot_data = (
                    bin_speeds,
                    bin_eff,
                    bin_pwr,
                    raw_speeds,
                    raw_eff,
                    raw_pwr,
                    power_result.best_range_speed_kmh,
                    power_result.best_endurance_speed_kmh,
                )

        # Determine if airspeed was used (wind-agnostic)
        use_airspeed = bool(log_data.airspeed)

        plots = visualizer.generate_all_plots(
            roll_data=roll_data,
            pitch_data=pitch_data,
            atrp_data=log_data.atrp,
            roll_steps=roll_steps,
            pitch_steps=pitch_steps,
            roll_metrics=roll_metrics,
            pitch_metrics=pitch_metrics,
            power_curve_data=power_curve_plot_data,
            use_airspeed=use_airspeed,
        )

        if plots:
            print(f"\nPlots saved to: {args.output}/")
            for name, path in plots.items():
                print(f"  - {os.path.basename(path)}")

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
