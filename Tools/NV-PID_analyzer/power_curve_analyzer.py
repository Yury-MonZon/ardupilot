#!/usr/bin/env python3
"""
Power Curve Analyzer

Analyzes power efficiency vs airspeed to find optimal cruise speeds:
- Best Range Speed: minimum Wh/km (maximum distance per battery)
- Best Endurance Speed: minimum power (maximum flight time)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d


@dataclass
class PowerDataPoint:
    """Synchronized power/speed data point"""

    time_s: float
    speed_mps: float
    power_w: float
    efficiency_wh_per_km: float
    climb_rate_mps: float
    altitude_m: float


@dataclass
class PowerCurveResult:
    """Result of power curve analysis"""

    best_range_speed_mps: float
    best_range_speed_kmh: float
    best_range_efficiency_wh_per_km: float
    best_range_power_w: float
    best_endurance_speed_mps: float
    best_endurance_speed_kmh: float
    best_endurance_power_w: float
    best_endurance_efficiency_wh_per_km: float
    total_data_points: int
    valid_data_points: int
    speed_range_mps: Tuple[float, float]
    speed_range_kmh: Tuple[float, float]
    efficiency_range_wh_per_km: Tuple[float, float]


class PowerCurveAnalyzer:
    """Analyze power curve from flight logs"""

    MIN_ALTITUDE_M = 10.0
    MIN_CURRENT_A = 0.5
    SPEED_BIN_SIZE_KMH = 2.0
    MIN_SAMPLES_PER_BIN = 50

    def __init__(self, airspeed_min_mps: float = 10.0, climb_limit_mps: float = 0.5):
        self.airspeed_min = airspeed_min_mps
        self.climb_limit = climb_limit_mps

    def extract_power_data(
        self,
        battery_data: List,
        gps_data: List,
        baro_data: List,
        airspeed_data: Optional[List] = None,
    ) -> List[PowerDataPoint]:
        """
        Synchronize and merge battery, GPS, and baro data.
        Uses airspeed (wind-agnostic) if available, otherwise falls back to GPS groundspeed.
        """
        if not battery_data or not gps_data:
            return []

        bat_times = np.array([b.time_s for b in battery_data])
        bat_volt = np.array([b.voltage for b in battery_data])
        bat_curr = np.array([b.current for b in battery_data])

        gps_times = np.array([g.time_s for g in gps_data])
        gps_speed = np.array([g.speed_mps for g in gps_data])
        gps_climb = np.array([g.climb_rate_mps for g in gps_data])

        if baro_data:
            baro_times = np.array([b.time_s for b in baro_data])
            baro_alt = np.array([b.altitude_m for b in baro_data])
            baro_climb = np.array([b.climb_rate_mps for b in baro_data])
        else:
            baro_times = np.array([])
            baro_alt = np.array([])
            baro_climb = np.array([])

        data_points = []

        if len(baro_times) > 2:
            climb_interp = interp1d(
                baro_times, baro_climb, bounds_error=False, fill_value=0
            )
            alt_interp = interp1d(
                baro_times, baro_alt, bounds_error=False, fill_value=100.0
            )
        else:
            climb_interp = interp1d(
                gps_times, gps_climb, bounds_error=False, fill_value=0
            )
            alt_interp = lambda t: 100.0

        # Use airspeed if available (wind-agnostic), otherwise use GPS groundspeed
        use_airspeed = airspeed_data is not None and len(airspeed_data) > 0
        if use_airspeed:
            airspeed_times = np.array([a.time_s for a in airspeed_data])
            airspeed = np.array([a.airspeed_mps for a in airspeed_data])
            speed_interp = interp1d(
                airspeed_times, airspeed, bounds_error=False, fill_value=0
            )
        else:
            speed_interp = interp1d(
                gps_times, gps_speed, bounds_error=False, fill_value=0
            )

        for i in range(len(bat_times)):
            t = bat_times[i]
            speed = float(speed_interp(t))
            climb = float(climb_interp(t))
            alt = float(alt_interp(t))
            power = bat_volt[i] * bat_curr[i]

            if speed > 0.1:
                efficiency = power / (speed * 3.6)
            else:
                efficiency = float("inf")

            data_points.append(
                PowerDataPoint(
                    time_s=t,
                    speed_mps=speed,
                    power_w=power,
                    efficiency_wh_per_km=efficiency,
                    climb_rate_mps=climb,
                    altitude_m=alt,
                )
            )

        return data_points

    def filter_data(self, data: List[PowerDataPoint]) -> List[PowerDataPoint]:
        """
        Filter data to include only stable cruise conditions.

        Filters:
        - Speed >= AIRSPEED_MIN
        - |climb_rate| < climb_limit (default 0.5 m/s, level flight)
        - Altitude > 10m (not ground handling)
        - Current > 0.5A (motor running, not gliding)
        """
        filtered = []
        reject_speed = 0
        reject_climb = 0
        reject_alt = 0
        reject_power = 0
        reject_eff = 0

        for dp in data:
            if dp.speed_mps < self.airspeed_min:
                reject_speed += 1
                continue
            if abs(dp.climb_rate_mps) > self.climb_limit:
                reject_climb += 1
                continue
            if dp.altitude_m < self.MIN_ALTITUDE_M:
                reject_alt += 1
                continue
            if dp.power_w < self.MIN_CURRENT_A * 12:
                reject_power += 1
                continue
            if not np.isfinite(dp.efficiency_wh_per_km):
                reject_eff += 1
                continue
            filtered.append(dp)

        print(f"  Filter stats:")
        print(f"    Input: {len(data)} points")
        print(f"    Rejected (speed < {self.airspeed_min} m/s): {reject_speed}")
        print(f"    Rejected (|climb| > {self.climb_limit} m/s): {reject_climb}")
        print(f"    Rejected (alt < {self.MIN_ALTITUDE_M}m): {reject_alt}")
        print(f"    Rejected (power < {self.MIN_CURRENT_A * 12}W): {reject_power}")
        print(f"    Rejected (infinite eff): {reject_eff}")
        print(f"    Valid: {len(filtered)} points")

        return filtered

    def analyze(self, data: List[PowerDataPoint]) -> Optional[PowerCurveResult]:
        """
        Analyze power curve and find optimal speeds using curve fitting.
        
        Detects boundary conditions where minimum falls at data edge (unreliable).
        """
        filtered = self.filter_data(data)

        if len(filtered) < 100:
            print(f"  Insufficient valid points: {len(filtered)} < 100")
            return None

        speeds_mps = np.array([dp.speed_mps for dp in filtered])
        powers_w = np.array([dp.power_w for dp in filtered])
        efficiencies = np.array([dp.efficiency_wh_per_km for dp in filtered])

        speeds_kmh = speeds_mps * 3.6
        speed_min = speeds_kmh.min()
        speed_max = speeds_kmh.max()
        speed_range = np.linspace(speed_min, speed_max, 1000)
        boundary_threshold = 0.05  # 5% from edge = boundary

        best_range_speed_kmh = None
        best_endurance_speed_kmh = None
        best_range_speed_mps = None
        best_endurance_speed_mps = None
        best_range_efficiency = None
        best_range_power = None
        best_endurance_power = None
        best_endurance_efficiency = None
        used_binned = False

        # Fit polynomial curves to ALL data
        # Efficiency vs Speed: quadratic (U-shaped curve)
        try:
            eff_coeffs = np.polyfit(speeds_kmh, efficiencies, deg=2)
            eff_fit = np.poly1d(eff_coeffs)

            # Find minimum of efficiency curve
            eff_curve = eff_fit(speed_range)
            best_range_idx = np.argmin(eff_curve)
            best_range_speed_kmh = speed_range[best_range_idx]
            best_range_efficiency = eff_curve[best_range_idx]
            best_range_speed_mps = best_range_speed_kmh / 3.6

            # Check if best range is at boundary (not expected - should be in middle)
            range_boundary_pos = (best_range_speed_kmh - speed_min) / (speed_max - speed_min)
            if range_boundary_pos < boundary_threshold:
                print(f"  Warning: Best range speed at low-speed boundary ({range_boundary_pos:.1%} of range)")
                print(f"           True optimum may be below observed speed range")
                print(f"           Try lowering AIRSPEED_MIN or flying slower to find true best range")
            elif range_boundary_pos > (1 - boundary_threshold):
                print(f"  Warning: Best range speed at high-speed boundary ({range_boundary_pos:.1%} of range)")
                print(f"           True optimum may be above observed speed range")

            # Get power at best range speed from power fit
            power_coeffs = np.polyfit(speeds_kmh, powers_w, deg=3)
            power_fit = np.poly1d(power_coeffs)
            best_range_power = power_fit(best_range_speed_kmh)

            # Power vs Speed: cubic (minimum at low speed, increases with speed)
            pwr_curve = power_fit(speed_range)
            best_endurance_idx = np.argmin(pwr_curve)
            best_endurance_speed_kmh = speed_range[best_endurance_idx]
            best_endurance_power = pwr_curve[best_endurance_idx]
            best_endurance_speed_mps = best_endurance_speed_kmh / 3.6
            best_endurance_efficiency = eff_fit(best_endurance_speed_kmh)

            # Note: Best endurance at low-speed boundary is EXPECTED (min power at min speed)
            # Only warn if it's at the high-speed boundary (unexpected)
            endurance_boundary_pos = (best_endurance_speed_kmh - speed_min) / (speed_max - speed_min)
            if endurance_boundary_pos > (1 - boundary_threshold):
                print(f"  Warning: Best endurance speed at high-speed boundary ({endurance_boundary_pos:.1%} of range)")
                print(f"           This is unusual - min power should be at low speed")

            # Sanity check: best range should be >= best endurance
            if best_range_speed_kmh < best_endurance_speed_kmh:
                print(f"  Warning: Best range ({best_range_speed_kmh:.1f} km/h) < Best endurance ({best_endurance_speed_kmh:.1f} km/h)")
                print(f"           This is physically unlikely - using binned data instead")
                raise ValueError("Range < Endurance, using binned fallback")

        except Exception as e:
            # Fallback to binned approach if curve fitting fails or gives bad results
            if not isinstance(e, ValueError):
                print(f"  Curve fit failed, using binned data")
            used_binned = True
            
            speed_bins = np.arange(
                speeds_kmh.min(),
                speeds_kmh.max() + self.SPEED_BIN_SIZE_KMH,
                self.SPEED_BIN_SIZE_KMH,
            )

            bin_efficiency = []
            bin_power = []
            bin_speed_mps = []

            for i in range(len(speed_bins) - 1):
                mask = (speeds_kmh >= speed_bins[i]) & (speeds_kmh < speed_bins[i + 1])
                if np.sum(mask) >= self.MIN_SAMPLES_PER_BIN:
                    bin_efficiency.append(np.mean(efficiencies[mask]))
                    bin_power.append(np.mean(powers_w[mask]))
                    bin_speed_mps.append(np.mean(speeds_mps[mask]))

            if len(bin_efficiency) < 5:
                return None

            bin_efficiency = np.array(bin_efficiency)
            bin_power = np.array(bin_power)
            bin_speed_mps = np.array(bin_speed_mps)
            bin_speed_kmh = bin_speed_mps * 3.6

            best_range_idx = np.argmin(bin_efficiency)
            best_range_speed_mps = bin_speed_mps[best_range_idx]
            best_range_speed_kmh = bin_speed_kmh[best_range_idx]
            best_range_efficiency = bin_efficiency[best_range_idx]
            best_range_power = bin_power[best_range_idx]

            best_endurance_idx = np.argmin(bin_power)
            best_endurance_speed_mps = bin_speed_mps[best_endurance_idx]
            best_endurance_speed_kmh = bin_speed_kmh[best_endurance_idx]
            best_endurance_power = bin_power[best_endurance_idx]
            best_endurance_efficiency = bin_efficiency[best_endurance_idx]

        return PowerCurveResult(
            best_range_speed_mps=best_range_speed_mps,
            best_range_speed_kmh=best_range_speed_kmh,
            best_range_efficiency_wh_per_km=best_range_efficiency,
            best_range_power_w=best_range_power,
            best_endurance_speed_mps=best_endurance_speed_mps,
            best_endurance_speed_kmh=best_endurance_speed_kmh,
            best_endurance_power_w=best_endurance_power,
            best_endurance_efficiency_wh_per_km=best_endurance_efficiency,
            total_data_points=len(data),
            valid_data_points=len(filtered),
            speed_range_mps=(float(speeds_mps.min()), float(speeds_mps.max())),
            speed_range_kmh=(float(speeds_kmh.min()), float(speeds_kmh.max())),
            efficiency_range_wh_per_km=(
                float(efficiencies.min()),
                float(efficiencies.max()),
            ),
        )

    def get_binned_data(
        self, data: List[PowerDataPoint]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get binned data and raw data for plotting.

        Returns (bin_centers, bin_efficiencies, bin_powers, raw_speeds, raw_efficiencies, raw_powers)
        """
        filtered = self.filter_data(data)

        if len(filtered) < 50:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        speeds_mps = np.array([dp.speed_mps for dp in filtered])
        powers_w = np.array([dp.power_w for dp in filtered])
        efficiencies = np.array([dp.efficiency_wh_per_km for dp in filtered])
        speeds_kmh = speeds_mps * 3.6

        # Binned data
        speed_bins = np.arange(
            speeds_kmh.min(),
            speeds_kmh.max() + self.SPEED_BIN_SIZE_KMH,
            self.SPEED_BIN_SIZE_KMH,
        )
        bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

        bin_efficiency = []
        bin_power = []

        for i in range(len(speed_bins) - 1):
            mask = (speeds_kmh >= speed_bins[i]) & (speeds_kmh < speed_bins[i + 1])
            if np.sum(mask) >= 5:
                bin_efficiency.append(np.mean(efficiencies[mask]))
                bin_power.append(np.mean(powers_w[mask]))
            else:
                bin_efficiency.append(np.nan)
                bin_power.append(np.nan)

        return (
            bin_centers,
            np.array(bin_efficiency),
            np.array(bin_power),
            speeds_kmh,
            efficiencies,
            powers_w,
        )


if __name__ == "__main__":
    import sys
    from log_parser import LogParser

    if len(sys.argv) < 2:
        print("Usage: python power_curve_analyzer.py <logfile.bin>")
        sys.exit(1)

    parser = LogParser(sys.argv[1])
    log_data = parser.parse()

    airspeed_min = log_data.params.get("AIRSPEED_MIN", 10.0)
    analyzer = PowerCurveAnalyzer(airspeed_min_mps=airspeed_min, climb_limit_mps=0.5)

    # Use airspeed data if available (wind-agnostic), otherwise GPS groundspeed
    data = analyzer.extract_power_data(
        log_data.battery, log_data.gps, log_data.baro, log_data.airspeed
    )
    result = analyzer.analyze(data)

    # Indicate which speed source was used
    speed_source = "Airspeed (wind-agnostic)" if log_data.airspeed else "GPS Groundspeed"

    if result:
        print(f"\nPOWER CURVE ANALYSIS")
        print("=" * 50)
        print(f"Speed Source: {speed_source}")
        print(
            f"Best Range Speed: {result.best_range_speed_kmh:.1f} km/h ({result.best_range_speed_mps:.1f} m/s) (data max: {result.speed_range_mps[1]:.1f} m/s)"
        )
        print(
            f"  Efficiency: {result.best_range_efficiency_wh_per_km:.3f} Wh/km (maximum distance per battery)"
        )
        print(f"  Power: {result.best_range_power_w:.0f} W")
        print(
            f"\nBest Endurance Speed: {result.best_endurance_speed_kmh:.1f} km/h ({result.best_endurance_speed_mps:.1f} m/s) (data min: {result.speed_range_mps[0]:.1f} m/s)"
        )
        print(f"  Power: {result.best_endurance_power_w:.0f} W (maximum flight time)")
        print(f"  Efficiency: {result.best_endurance_efficiency_wh_per_km:.3f} Wh/km")
        print(f"\nData quality:")
        print(
            f"  Valid samples: {result.valid_data_points} / {result.total_data_points}"
        )
        print(
            f"  Speed range: {result.speed_range_kmh[0]:.1f} - {result.speed_range_kmh[1]:.1f} km/h"
        )
        print(
            f"  Efficiency range: {result.efficiency_range_wh_per_km[0]:.3f} - {result.efficiency_range_wh_per_km[1]:.3f} Wh/km"
        )
    else:
        print("\nInsufficient data for power curve analysis")
