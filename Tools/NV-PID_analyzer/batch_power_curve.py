#!/usr/bin/env python3
"""
Batch Power Curve Analyzer

Analyzes power curve performance across multiple log files from different flights.
Combines data from all logs to get a comprehensive power curve analysis.

Usage:
    python batch_power_curve.py <folder_path>
    
Options:
    --airspeed-min FLOAT    Override AIRSPEED_MIN parameter (default: use param from logs)
    --min-logs INT          Minimum number of valid logs required (default: 1)
    --no-plot               Skip generating plots
    --output DIR            Output directory for plots (default: ./plots)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from log_parser import LogParser
from power_curve_analyzer import PowerCurveAnalyzer, PowerDataPoint, PowerCurveResult


@dataclass
class LogSummary:
    """Summary of a single log file analysis"""
    filepath: str
    duration_s: float
    airspeed_samples: int
    battery_samples: int
    gps_samples: int
    valid_power_points: int
    has_airspeed: bool


@dataclass
class BatchResult:
    """Result of batch power curve analysis"""
    total_logs: int
    valid_logs: int
    total_power_points: int
    combined_result: Optional[PowerCurveResult]
    log_summaries: List[LogSummary]
    all_power_data: List[PowerDataPoint]


def find_bin_files(folder: str) -> List[str]:
    """Recursively find all .bin files in folder and subfolders"""
    bin_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    return sorted(bin_files)


def check_airspeed_enabled(filepath: str) -> Tuple[bool, Optional[float]]:
    """
    Quick check if ARSPD_USE=1 and ARSP messages exist in log file.
    Only reads PARM messages and samples for ARSP - doesn't parse entire log.
    
    Returns:
        Tuple of (airspeed_enabled, airspeed_min) or (False, None) if not found/error
    """
    try:
        from pymavlink import mavutil
        
        mlog = mavutil.mavlink_connection(filepath)
        arspd_use = None
        airspeed_min = None
        has_arsp_messages = False
        
        # First: Only read PARM messages to get ARSPD_USE
        while True:
            msg = mlog.recv_match(type=['PARM'])
            if msg is None:
                break
            
            name = getattr(msg, 'Name', '')
            value = getattr(msg, 'Value', None)
            
            if name == 'ARSPD_USE':
                arspd_use = value
            elif name == 'AIRSPEED_MIN':
                airspeed_min = value
            
            # Early exit if ARSPD_USE=0
            if arspd_use is not None and arspd_use == 0:
                return False, None
        
        # ARSPD_USE defaults to 1 if not present
        if arspd_use is None:
            arspd_use = 1
        
        # If ARSPD_USE=0, skip
        if int(arspd_use) != 1:
            return False, None
        
        # Second: Quick scan for ARSP messages (sample, don't read all)
        # Reset and scan first 5000 messages looking for ARSP
        mlog = mavutil.mavlink_connection(filepath)
        sample_count = 0
        max_samples = 5000
        
        while sample_count < max_samples:
            msg = mlog.recv_match()
            if msg is None:
                break
            sample_count += 1
            if msg.get_type() == 'ARSP':
                has_arsp_messages = True
                break
        
        return has_arsp_messages, airspeed_min if has_arsp_messages else None
        
    except Exception as e:
        return False, None


def extract_power_data_from_log(
    filepath: str,
    airspeed_min_override: Optional[float] = None,
    require_airspeed: bool = True,
    cached_airspeed_min: Optional[float] = None,
    climb_limit: float = 0.5
) -> Tuple[Optional[List[PowerDataPoint]], Optional[LogSummary]]:
    """
    Extract power data from a single log file.
    
    Returns:
        Tuple of (power_data, log_summary) or (None, None) if log is invalid
    """
    try:
        parser = LogParser(filepath)
        log_data = parser.parse()
        
        # Check if log has required data
        has_battery = len(log_data.battery) > 0
        has_gps = len(log_data.gps) > 0
        has_airspeed = len(log_data.airspeed) > 0
        
        if not has_battery or not has_gps:
            return None, None
        
        # Check if airspeed sensor is enabled in params
        arspd_use = log_data.params.get("ARSPD_USE", 0)
        airspeed_enabled = int(arspd_use) == 1
        
        # Skip logs without airspeed sensor enabled
        if require_airspeed and (not has_airspeed or not airspeed_enabled):
            return None, None
        
        # Determine airspeed minimum
        if airspeed_min_override is not None:
            airspeed_min = airspeed_min_override
        elif cached_airspeed_min is not None:
            airspeed_min = cached_airspeed_min
        else:
            airspeed_min = log_data.params.get("AIRSPEED_MIN", 10.0)
        
        # Extract power data
        analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=climb_limit)
        power_data = analyzer.extract_power_data(
            log_data.battery,
            log_data.gps,
            log_data.baro,
            log_data.airspeed if has_airspeed else None
        )
        
        # Create summary
        summary = LogSummary(
            filepath=filepath,
            duration_s=log_data.duration_s,
            airspeed_samples=len(log_data.airspeed),
            battery_samples=len(log_data.battery),
            gps_samples=len(log_data.gps),
            valid_power_points=len(power_data),
            has_airspeed=has_airspeed and airspeed_enabled
        )
        
        return power_data, summary
        
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None, None


def analyze_batch(
    bin_files: List[str],
    airspeed_min_override: Optional[float] = None,
    min_logs: int = 1,
    require_airspeed: bool = True,
    climb_limit: float = 0.5
) -> BatchResult:
    """
    Analyze power curve from multiple log files.
    
    Combines all power data points and runs a single analysis on the combined dataset.
    """
    print(f"\nBATCH POWER CURVE ANALYSIS")
    print("=" * 80)
    print(f"Found {len(bin_files)} .bin files")
    if require_airspeed:
        print("Filter: Only logs with ARSPD_USE=1 and ARSP messages will be analyzed")
    print()
    
    # First pass: Quick check ARSPD_USE parameter and ARSP messages for all files
    airspeed_mins = {}
    if require_airspeed:
        print("Phase 1: Checking ARSPD_USE=1 and ARSP messages...")
        eligible_files = []
        
        for filepath in bin_files:
            enabled, airspeed_min = check_airspeed_enabled(filepath)
            if enabled:
                eligible_files.append(filepath)
                if airspeed_min is not None:
                    airspeed_mins[filepath] = airspeed_min
        
        print(f"  Eligible logs (ARSPD_USE=1 + ARSP messages): {len(eligible_files)}/{len(bin_files)}")
        print()
        bin_files = eligible_files
    
    all_power_data: List[PowerDataPoint] = []
    log_summaries: List[LogSummary] = []
    total_logs = len(bin_files)
    valid_logs = 0
    skipped_no_airspeed = 0
    
    for i, filepath in enumerate(bin_files, 1):
        print(f"[{i}/{len(bin_files)}] {os.path.basename(filepath)}")
        
        # Get cached airspeed_min from quick check
        cached_min = airspeed_mins.get(filepath) if require_airspeed else None
        
        power_data, summary = extract_power_data_from_log(
            filepath, airspeed_min_override, require_airspeed, cached_min, climb_limit
        )
        
        if power_data is not None and len(power_data) > 0:
            all_power_data.extend(power_data)
            log_summaries.append(summary)
            valid_logs += 1
            print(f"    ✓ {len(power_data)} power points")
        else:
            if require_airspeed and summary is None:
                skipped_no_airspeed += 1
                print(f"    ✗ Skipped (no battery/GPS data)")
            else:
                print(f"    ✗ Skipped (no battery/GPS data)")
    
    print()
    if require_airspeed:
        print(f"Skipped (no airspeed): {skipped_no_airspeed}")
    print(f"Valid logs: {valid_logs}/{total_logs}")
    print(f"Total power data points: {len(all_power_data):,}")
    print()
    
    if valid_logs < min_logs:
        print(f"Error: Need at least {min_logs} valid logs, got {valid_logs}")
        return BatchResult(
            total_logs=total_logs,
            valid_logs=valid_logs,
            total_power_points=len(all_power_data),
            combined_result=None,
            log_summaries=log_summaries,
            all_power_data=all_power_data
        )
    
    # Determine airspeed minimum for analysis
    if airspeed_min_override is not None:
        airspeed_min = airspeed_min_override
    elif log_summaries:
        # Use the most common AIRSPEED_MIN from logs
        airspeed_min = 10.0  # default
    else:
        airspeed_min = 10.0
    
    # Run analysis on combined data
    print("=" * 80)
    print("COMBINED POWER CURVE ANALYSIS")
    print("=" * 80)

    analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=climb_limit)
    combined_result = analyzer.analyze(all_power_data)
    
    return BatchResult(
        total_logs=total_logs,
        valid_logs=valid_logs,
        total_power_points=len(all_power_data),
        combined_result=combined_result,
        log_summaries=log_summaries,
        all_power_data=all_power_data
    )


def print_batch_result(result: BatchResult, airspeed_min: float, require_airspeed: bool = True):
    """Print batch analysis results"""
    if result.combined_result is None:
        print("\nInsufficient data for combined power curve analysis")
        print("  (need stable cruise conditions across logs)")
        return

    r = result.combined_result
    speed_min_mps = r.speed_range_mps[0]
    speed_max_mps = r.speed_range_mps[1]

    print(f"\nSpeed Source: {'Airspeed (wind-agnostic)' if require_airspeed else 'Mixed (Airspeed preferred, GPS fallback)'}")
    print()
    print(f"BEST RANGE SPEED: {r.best_range_speed_kmh:.1f} km/h "
          f"({r.best_range_speed_mps:.1f} m/s) "
          f"(data max: {speed_max_mps:.1f} m/s)")
    print(f"  Efficiency: {r.best_range_efficiency_wh_per_km:.3f} Wh/km "
          f"(maximum distance per battery)")
    print(f"  Power: {r.best_range_power_w:.0f} W")
    print()
    print(f"BEST ENDURANCE SPEED: {r.best_endurance_speed_kmh:.1f} km/h "
          f"({r.best_endurance_speed_mps:.1f} m/s) "
          f"(data min: {speed_min_mps:.1f} m/s)")
    print(f"  Power: {r.best_endurance_power_w:.0f} W (maximum flight time)")
    print(f"  Efficiency: {r.best_endurance_efficiency_wh_per_km:.3f} Wh/km")
    print()
    print(f"DATA QUALITY:")
    print(f"  Logs analyzed: {result.valid_logs}")
    print(f"  Total power points: {r.total_data_points:,}")
    print(f"  Valid after filtering: {r.valid_data_points:,} "
          f"({100*r.valid_data_points/r.total_data_points:.1f}%)")
    print(f"  Speed range: {r.speed_range_kmh[0]:.1f} - {r.speed_range_kmh[1]:.1f} km/h")
    print(f"  Efficiency range: {r.efficiency_range_wh_per_km[0]:.3f} - "
          f"{r.efficiency_range_wh_per_km[1]:.3f} Wh/km")
    
    # Print per-log summary
    print()
    print("PER-LOG SUMMARY:")
    print("-" * 80)
    print(f"{'Log File':<50} {'Points':>10} {'Airspeed':>10} {'Duration':>10}")
    print("-" * 80)
    
    for summary in sorted(result.log_summaries, key=lambda x: x.valid_power_points, reverse=True):
        filename = os.path.basename(summary.filepath)
        if len(filename) > 48:
            filename = filename[:45] + "..."
        airspeed_str = "✓" if summary.has_airspeed else "✗"
        duration_str = f"{summary.duration_s/60:.1f} min"
        print(f"{filename:<50} {summary.valid_power_points:>10,} {airspeed_str:>10} {duration_str:>10}")
    
    print("-" * 80)


def generate_batch_plot(
    result: BatchResult,
    analyzer: PowerCurveAnalyzer,
    output_dir: str
) -> Optional[str]:
    """Generate a plot for the batch analysis"""
    if result.combined_result is None:
        return None
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Get binned data
        (bin_centers, bin_efficiencies, bin_powers, 
         raw_speeds, raw_efficiencies, raw_powers) = analyzer.get_binned_data(
            result.all_power_data
        )
        
        if len(bin_centers) == 0:
            return None
        
        # Filter NaN
        valid = ~np.isnan(bin_efficiencies) & ~np.isnan(bin_powers)
        bin_speeds = bin_centers[valid]
        bin_eff = bin_efficiencies[valid]
        bin_pwr = bin_powers[valid]
        
        # Fit curves
        try:
            eff_coeffs = np.polyfit(raw_speeds, raw_efficiencies, deg=2)
            eff_fit = np.poly1d(eff_coeffs)
            power_coeffs = np.polyfit(raw_speeds, raw_powers, deg=3)
            power_fit = np.poly1d(power_coeffs)
            
            speed_range = np.linspace(raw_speeds.min(), raw_speeds.max(), 200)
            eff_curve = eff_fit(speed_range)
            pwr_curve = power_fit(speed_range)
            has_fit = True
        except Exception:
            has_fit = False
        
        r = result.combined_result
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Efficiency plot
        ax1.scatter(raw_speeds, raw_efficiencies, alpha=0.1, s=2, c="blue",
                   label=f"Raw data (n={len(raw_speeds):,})")
        ax1.plot(bin_speeds, bin_eff, "b-o", linewidth=2, markersize=6,
                label="Binned average")
        if has_fit:
            ax1.plot(speed_range, eff_curve, "g-", linewidth=2.5, label="Quadratic fit")
        ax1.axvline(r.best_range_speed_kmh, color="red", linestyle="--", linewidth=2,
                   label=f"Best Range: {r.best_range_speed_kmh:.1f} km/h")
        ax1.set_ylabel("Efficiency (Wh/km)")
        ax1.set_title(f"Batch Power Curve Analysis - {result.valid_logs} logs, "
                     f"{result.total_power_points:,} points")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Power plot
        ax2.scatter(raw_speeds, raw_powers, alpha=0.1, s=2, c="red",
                   label=f"Raw data (n={len(raw_speeds):,})")
        ax2.plot(bin_speeds, bin_pwr, "r-o", linewidth=2, markersize=6,
                label="Binned average")
        if has_fit:
            ax2.plot(speed_range, pwr_curve, "orange", linewidth=2.5, label="Cubic fit")
        ax2.axvline(r.best_endurance_speed_kmh, color="green", linestyle="--", linewidth=2,
                   label=f"Best Endurance: {r.best_endurance_speed_kmh:.1f} km/h")
        ax2.set_xlabel("Speed (km/h)")
        ax2.set_ylabel("Power (W)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "batch_power_curve.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch Power Curve Analyzer - Analyze multiple log files"
    )
    parser.add_argument(
        "folder",
        help="Folder containing .bin log files (searches recursively)"
    )
    parser.add_argument(
        "--airspeed-min",
        type=float,
        default=None,
        help="Override AIRSPEED_MIN parameter (default: use param from logs)"
    )
    parser.add_argument(
        "--min-logs",
        type=int,
        default=1,
        help="Minimum number of valid logs required (default: 1)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./plots",
        help="Output directory for plots (default: ./plots)"
    )
    parser.add_argument(
        "--allow-no-airspeed",
        action="store_true",
        help="Allow logs without ARSPD_USE=1 or ARSP messages (uses GPS groundspeed)"
    )
    parser.add_argument(
        "--climb-limit",
        type=float,
        default=0.5,
        help="Max climb rate for stable flight filter (default: 0.5 m/s)"
    )
    
    args = parser.parse_args()
    
    # Find all .bin files
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)
    
    bin_files = find_bin_files(args.folder)
    
    if not bin_files:
        print(f"No .bin files found in: {args.folder}")
        sys.exit(1)
    
    # Run batch analysis
    require_airspeed = not args.allow_no_airspeed
    result = analyze_batch(
        bin_files,
        airspeed_min_override=args.airspeed_min,
        min_logs=args.min_logs,
        require_airspeed=require_airspeed,
        climb_limit=args.climb_limit
    )

    # Determine airspeed min for display
    airspeed_min = args.airspeed_min if args.airspeed_min is not None else 10.0

    # Print results
    print_batch_result(result, airspeed_min, require_airspeed)

    # Generate plot
    if not args.no_plot and result.combined_result is not None:
        os.makedirs(args.output, exist_ok=True)

        analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=args.climb_limit)
        plot_path = generate_batch_plot(result, analyzer, args.output)
        
        if plot_path:
            print(f"\nPlot saved to: {plot_path}")
    
    print()


if __name__ == "__main__":
    main()
