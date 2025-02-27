#!/usr/bin/env python3
"""
ArduPlane PID Analysis Tool

This script analyzes ArduPlane log files to evaluate PID controller performance
and suggest tuning adjustments based on observed behavior.

Usage:
    python analyze_arduplane_pids.py path/to/logfile.bin

Requirements:
    - pymavlink
    - numpy
    - matplotlib (for visualization)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Analyze ArduPlane logs and suggest PID adjustments')
    parser.add_argument('logfile', help='Path to the .bin log file')
    parser.add_argument('--plot', action='store_true', help='Generate plots of PID performance')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed analysis')
    parser.add_argument('--output', '-o', help='Output file for recommendations')
    return parser.parse_args()

class PIDAnalyzer:
    def validate_logfile(self):
        """Validate the log file exists and contains required messages"""
        if not os.path.exists(self.logfile):
            raise FileNotFoundError(f"Log file not found: {self.logfile}")
        
        # Check file size
        if os.path.getsize(self.logfile) == 0:
            raise ValueError("Log file is empty")
        
        # Verify we can read the file
        try:
            mlog = mavutil.mavlink_connection(self.logfile)
            msg = mlog.recv_match()
            if msg is None:
                raise ValueError("No valid MAVLink messages found in log file")
        except Exception as e:
            raise ValueError(f"Failed to read log file: {str(e)}")
        
        return True

    def __init__(self, logfile):
        """Initialize with validation"""
        self.logfile = logfile
        self.validate_logfile()
        self.mlog = mavutil.mavlink_connection(logfile)
        
        # Data storage
        self.roll_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.pitch_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.yaw_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.nav_data = {'time': [], 'aspd': [], 'gspd': [], 'target_aspd': []}
        
        # Thresholds for analysis
        self.oscillation_threshold = 0.2  # Normalized oscillation detection
        self.overshoot_threshold = 0.3    # 30% overshoot
        self.steady_state_error_threshold = 0.1  # 10% steady state error
        self.response_time_threshold = 1.0  # seconds
        
        # Flight modes
        self.flight_modes = []
        
        # Parameters from log
        self.parameters = {}
        
        # Define valid flight modes for analysis
        self.valid_modes = ['CRUISE', 'FBWA', 'RTL']
        
    def load_data(self):
        """Load and process log data"""
        print(f"Loading data from {self.logfile}...")
        
        # Initialize lists (not numpy arrays) for data collection
        self.roll_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.pitch_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.yaw_data = {'time': [], 'target': [], 'actual': [], 'P': [], 'I': [], 'D': [], 'FF': [], 'Dmod': []}
        self.nav_data = {'time': [], 'aspd': [], 'gspd': [], 'target_aspd': []}
        self.flight_modes = []
        
        # Single pass to get all data
        while True:
            msg = self.mlog.recv_match(blocking=False)
            if msg is None:
                break
            
            msg_type = msg.get_type()
            
            # Get parameters
            if msg_type == 'PARM':
                self.parameters[msg.Name] = msg.Value
                continue
                
            # Get flight modes
            if msg_type == 'MODE':
                self.flight_modes.append([msg.TimeUS / 1e6, msg.Mode])
                
            # Get roll PID data
            elif msg_type == 'PIDR':
                self.roll_data['time'].append(msg.TimeUS / 1e6)
                self.roll_data['target'].append(msg.Tar)
                self.roll_data['actual'].append(msg.Act)
                self.roll_data['P'].append(msg.P)
                self.roll_data['I'].append(msg.I)
                self.roll_data['D'].append(msg.D)
                self.roll_data['FF'].append(msg.FF if hasattr(msg, 'FF') else 0)
                self.roll_data['Dmod'].append(msg.Dmod if hasattr(msg, 'Dmod') else 1.0)
                
            # Get pitch PID data
            elif msg_type == 'PIDP':
                self.pitch_data['time'].append(msg.TimeUS / 1e6)
                self.pitch_data['target'].append(msg.Tar)
                self.pitch_data['actual'].append(msg.Act)
                self.pitch_data['P'].append(msg.P)
                self.pitch_data['I'].append(msg.I)
                self.pitch_data['D'].append(msg.D)
                self.pitch_data['FF'].append(msg.FF if hasattr(msg, 'FF') else 0)
                self.pitch_data['Dmod'].append(msg.Dmod if hasattr(msg, 'Dmod') else 1.0)
                
            # Get yaw PID data
            elif msg_type == 'PIDY':
                self.yaw_data['time'].append(msg.TimeUS / 1e6)
                self.yaw_data['target'].append(msg.Tar)
                self.yaw_data['actual'].append(msg.Act)
                self.yaw_data['P'].append(msg.P)
                self.yaw_data['I'].append(msg.I)
                self.yaw_data['D'].append(msg.D)
                self.yaw_data['FF'].append(msg.FF if hasattr(msg, 'FF') else 0)
                self.yaw_data['Dmod'].append(msg.Dmod if hasattr(msg, 'Dmod') else 1.0)
                
            # Get navigation data
            elif msg_type == 'CTUN':
                self.nav_data['time'].append(msg.TimeUS / 1e6)
                self.nav_data['aspd'].append(msg.Aspd if hasattr(msg, 'Aspd') else 0)
                self.nav_data['gspd'].append(msg.GndSpd if hasattr(msg, 'GndSpd') else 0)
                self.nav_data['target_aspd'].append(msg.TAS if hasattr(msg, 'TAS') else 0)
        
        # Convert lists to numpy arrays after all data is collected
        self.flight_modes = np.array(self.flight_modes)
        
        # Convert all data lists to numpy arrays
        for axis in [self.roll_data, self.pitch_data, self.yaw_data, self.nav_data]:
            for key in axis:
                axis[key] = np.array(axis[key])
        
        print(f"Loaded {len(self.roll_data['time'])} roll data points")
        print(f"Loaded {len(self.pitch_data['time'])} pitch data points")
        print(f"Loaded {len(self.yaw_data['time'])} yaw data points")
        
    def is_valid_flight_condition(self, timestamp, airspeed):
        """Check if this is a valid flight condition for analysis"""
        if len(self.flight_modes) == 0:
            print("Warning: No flight modes recorded")
            return True
        
        # Find current flight mode
        mode_idx = np.searchsorted(self.flight_modes[:, 0], timestamp) - 1
        if mode_idx < 0:
            return True
        
        current_mode = self.flight_modes[mode_idx][1]
        
        # Check if mode is valid
        if current_mode not in self.valid_modes:
            return False
        
        # Check airspeed
        min_airspeed = 8.0  # m/s
        if airspeed < min_airspeed:
            return False
        
        return True

    def analyze_pid_performance(self, axis_data, axis_name):
        """Analyze PID performance for a specific axis"""
        analysis = {
            'has_data': False,
            'message': f"No {axis_name} PID data found in log",
            'oscillation_frequency': 0.0,
            'mean_error': 0.0,
            'max_error': 0.0,
            'steady_state_error_pct': 0.0,
            'steady_state_bias': 0.0,
            'overshoot_pct': 0.0,
            'p_contribution': 0.0,
            'i_contribution': 0.0,
            'd_contribution': 0.0,
            'ff_contribution': 0.0,
            'dmod_min': 1.0,
            'dmod_max': 1.0,
            'p_oscillation': False,
            'i_oscillation': False,
            'd_oscillation': False,
            'p_term_average': 0.0,
            'i_term_average': 0.0,
            'd_term_average': 0.0
        }

        if len(axis_data['time']) == 0:
            return analysis

        # Basic error calculations
        error = axis_data['actual'] - axis_data['target']
        analysis['has_data'] = True
        analysis['mean_error'] = float(np.mean(np.abs(error)))
        analysis['max_error'] = float(np.max(np.abs(error)))
        analysis['steady_state_bias'] = float(np.mean(error))

        # Calculate steady state error as percentage of target
        nonzero_targets = axis_data['target'][np.abs(axis_data['target']) > 0.1]
        if len(nonzero_targets) > 0:
            corresponding_error = error[np.abs(axis_data['target']) > 0.1]
            analysis['steady_state_error_pct'] = float(np.mean(corresponding_error / nonzero_targets) * 100)

        # Calculate overshoot with improved logic
        def calculate_overshoot(target, actual):
            error = actual - target
            max_overshoot = 0
            for i in range(1, len(error)-1):
                if abs(target[i]) > 0.1:  # Only consider significant commands
                    overshoot = abs(error[i] / target[i]) * 100
                    if overshoot > max_overshoot and overshoot < 300:  # Cap at 300% to avoid unrealistic values
                        max_overshoot = overshoot
            return max_overshoot

        analysis['overshoot_pct'] = calculate_overshoot(axis_data['target'], axis_data['actual'])

        # Calculate PID contributions
        total_terms = np.abs(axis_data['P']) + np.abs(axis_data['I']) + np.abs(axis_data['D']) + np.abs(axis_data['FF'])
        nonzero_total = total_terms > 0.01  # Increased threshold to avoid noise
        if np.any(nonzero_total):
            analysis['p_contribution'] = float(np.mean(np.abs(axis_data['P'][nonzero_total]) / total_terms[nonzero_total]) * 100)
            analysis['i_contribution'] = float(np.mean(np.abs(axis_data['I'][nonzero_total]) / total_terms[nonzero_total]) * 100)
            analysis['d_contribution'] = float(np.mean(np.abs(axis_data['D'][nonzero_total]) / total_terms[nonzero_total]) * 100)
            analysis['ff_contribution'] = float(np.mean(np.abs(axis_data['FF'][nonzero_total]) / total_terms[nonzero_total]) * 100)

        # Calculate average term values
        analysis['p_term_average'] = float(np.mean(np.abs(axis_data['P'])))
        analysis['i_term_average'] = float(np.mean(np.abs(axis_data['I'])))
        analysis['d_term_average'] = float(np.mean(np.abs(axis_data['D'])))

        # Improved oscillation detection
        def detect_oscillation(signal, min_amplitude=0.05):
            if len(signal) < 10:
                return False
            
            # Detrend the signal
            detrended = signal - np.mean(signal)
            
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
            
            if len(zero_crossings) < 4:
                return False
            
            # Check amplitude
            peaks = []
            for i in range(1, len(detrended)-1):
                if abs(detrended[i]) > min_amplitude:
                    if detrended[i] > detrended[i-1] and detrended[i] > detrended[i+1]:
                        peaks.append(detrended[i])
                    elif detrended[i] < detrended[i-1] and detrended[i] < detrended[i+1]:
                        peaks.append(detrended[i])
                    
            if len(peaks) < 4:
                return False
            
            # Check regularity
            peak_periods = np.diff(peaks)
            if np.std(peak_periods) / np.mean(peak_periods) < 0.5:
                return True
            
            return False

        # Calculate oscillation frequency with improved detection
        if len(error) > 10:
            detrended_error = error - np.mean(error)
            zero_crossings = np.where(np.diff(np.signbit(detrended_error)))[0]
            if len(zero_crossings) >= 4:
                periods = np.diff(axis_data['time'][zero_crossings])
                if len(periods) > 0 and np.std(periods) / np.mean(periods) < 0.5:
                    analysis['oscillation_frequency'] = float(1.0 / (2.0 * np.mean(periods)))

        # Detect oscillations in individual terms with appropriate thresholds
        analysis['p_oscillation'] = detect_oscillation(axis_data['P'], min_amplitude=0.05)
        analysis['i_oscillation'] = detect_oscillation(axis_data['I'], min_amplitude=0.02)
        analysis['d_oscillation'] = detect_oscillation(axis_data['D'], min_amplitude=0.05)

        # Calculate Dmod range with validation
        dmod_values = axis_data['Dmod']
        valid_dmod = dmod_values[(dmod_values > 0) & (dmod_values <= 1)]
        if len(valid_dmod) > 0:
            analysis['dmod_min'] = float(np.min(valid_dmod))
            analysis['dmod_max'] = float(np.max(valid_dmod))
        else:
            analysis['dmod_min'] = 1.0
            analysis['dmod_max'] = 1.0

        return analysis
        
    def analyze_dmod_range(self, axis_data, axis_name):
        """Analyze Dmod range during stable flight modes"""
        if 'Dmod' not in axis_data or len(axis_data['Dmod']) == 0:
            return None
            
        dmod_array = np.array(axis_data['Dmod'])
        
        # Only report if we see meaningful Dmod variation
        # Check if there's any significant deviation from 1.0
        if np.any(dmod_array < 0.95):  # Only report if Dmod actually reduced below 0.95
            dmod_min = np.min(dmod_array)
            dmod_max = np.max(dmod_array)
            
            # Skip reporting if:
            # 1. We're just seeing noise around 1.0
            # 2. We have default values (0.0 to 1.0)
            # 3. The variation is too small to be meaningful
            if (dmod_min < 0.95 and  # Must see actual reduction
                not (np.isclose(dmod_min, 0.0) and np.isclose(dmod_max, 1.0)) and  # Not default values
                (dmod_max - dmod_min) > 0.1):  # Must have meaningful variation
                return f"{axis_name} Dmod Range: {dmod_min:.2f} - {dmod_max:.2f}"
        
        return None

    def generate_recommendations(self, roll_analysis, pitch_analysis, yaw_analysis):
        """Generate PID tuning recommendations based on analysis"""
        recommendations = []
        
        def add_recommendation(param, current, suggested, reason):
            # Enforce minimum and maximum values
            # Limit reduction to 15% per iteration for safety
            min_value = current * 0.85  
            # Limit increase to 20% per iteration for safety
            max_value = current * 1.2   
            
            suggested = max(min_value, min(suggested, max_value))
            
            # Only add recommendation if change is significant (>5%)
            if abs(suggested - current) / current < 0.05:
                return
                
            # Check if we already have a recommendation for this parameter
            for rec in recommendations:
                if rec['param'] == param:
                    return
                
            recommendations.append({
                'param': param,
                'current': current,
                'suggested': suggested,
                'reason': reason
            })

        # Process roll axis
        if roll_analysis['has_data']:
            current_p = self.parameters.get('RLL_RATE_P', 0)
            current_i = self.parameters.get('RLL_RATE_I', 0)
            
            # P term adjustments
            p_reduction_needed = False
            p_reduction_factor = 1.0
            p_reason_parts = []
            
            if roll_analysis['overshoot_pct'] > self.overshoot_threshold * 100:
                p_reduction_needed = True
                # Scale reduction based on overshoot severity (max 15% reduction)
                p_reduction_factor = 1.0 - min(0.15, (roll_analysis['overshoot_pct'] - self.overshoot_threshold * 100) / 300)
                p_reason_parts.append(f"overshoot {roll_analysis['overshoot_pct']:.1f}%")
            
            if p_reduction_needed:
                suggested_p = current_p * p_reduction_factor
                reason = f"Reduce P: {' and '.join(p_reason_parts)}"
                add_recommendation('RLL_RATE_P', current_p, suggested_p, reason)
            
            # I term adjustments
            if abs(roll_analysis['steady_state_error_pct']) > self.steady_state_error_threshold * 100:
                # Scale I change based on error magnitude (max 15% change)
                i_change = 1.0 + min(0.15, abs(roll_analysis['steady_state_error_pct']) / 100) * (1 if roll_analysis['steady_state_error_pct'] > 0 else -1)
                suggested_i = current_i * i_change
                reason = f"{'Increase' if i_change > 1 else 'Decrease'} I: steady error {roll_analysis['steady_state_error_pct']:.1f}%"
                add_recommendation('RLL_RATE_I', current_i, suggested_i, reason)

        # Process pitch axis (similar logic)
        if pitch_analysis['has_data']:
            current_p = self.parameters.get('PTCH_RATE_P', 0)
            current_i = self.parameters.get('PTCH_RATE_I', 0)
            
            p_reduction_needed = False
            p_reduction_factor = 1.0
            p_reason_parts = []
            
            if pitch_analysis['overshoot_pct'] > self.overshoot_threshold * 100:
                p_reduction_needed = True
                p_reduction_factor = 1.0 - min(0.15, (pitch_analysis['overshoot_pct'] - self.overshoot_threshold * 100) / 300)
                p_reason_parts.append(f"overshoot {pitch_analysis['overshoot_pct']:.1f}%")
            
            if p_reduction_needed:
                suggested_p = current_p * p_reduction_factor
                reason = f"Reduce P: {' and '.join(p_reason_parts)}"
                add_recommendation('PTCH_RATE_P', current_p, suggested_p, reason)
            
            if abs(pitch_analysis['steady_state_error_pct']) > self.steady_state_error_threshold * 100:
                i_change = 1.0 + min(0.15, abs(pitch_analysis['steady_state_error_pct']) / 100) * (1 if pitch_analysis['steady_state_error_pct'] > 0 else -1)
                suggested_i = current_i * i_change
                reason = f"{'Increase' if i_change > 1 else 'Decrease'} I: steady error {pitch_analysis['steady_state_error_pct']:.1f}%"
                add_recommendation('PTCH_RATE_I', current_i, suggested_i, reason)

        return recommendations
        
    def plot_pid_data(self):
        """Generate plots of PID performance"""
        # Create two figures - one for PID data and one for navigation
        fig_pid, ((ax_roll, ax_pitch), (ax_yaw, ax_pid_components)) = plt.subplots(2, 2, figsize=(15, 10))
        fig_pid.suptitle('PID Controller Performance')
        
        # Plot roll data
        ax_roll.plot(self.roll_data['time'], self.roll_data['target'], 'b-', label='Target', alpha=0.7)
        ax_roll.plot(self.roll_data['time'], self.roll_data['actual'], 'r-', label='Actual', alpha=0.7)
        ax_roll.set_title('Roll Control')
        ax_roll.set_xlabel('Time (s)')
        ax_roll.set_ylabel('Angle (deg)')
        ax_roll.legend()
        ax_roll.grid(True)
        
        # Plot pitch data
        ax_pitch.plot(self.pitch_data['time'], self.pitch_data['target'], 'b-', label='Target', alpha=0.7)
        ax_pitch.plot(self.pitch_data['time'], self.pitch_data['actual'], 'r-', label='Actual', alpha=0.7)
        ax_pitch.set_title('Pitch Control')
        ax_pitch.set_xlabel('Time (s)')
        ax_pitch.set_ylabel('Angle (deg)')
        ax_pitch.legend()
        ax_pitch.grid(True)
        
        # Plot yaw data
        ax_yaw.plot(self.yaw_data['time'], self.yaw_data['target'], 'b-', label='Target', alpha=0.7)
        ax_yaw.plot(self.yaw_data['time'], self.yaw_data['actual'], 'r-', label='Actual', alpha=0.7)
        ax_yaw.set_title('Yaw Control')
        ax_yaw.set_xlabel('Time (s)')
        ax_yaw.set_ylabel('Angle (deg)')
        ax_yaw.legend()
        ax_yaw.grid(True)
        
        # Plot PID components for roll (as an example)
        ax_pid_components.plot(self.roll_data['time'], self.roll_data['P'], 'r-', label='P', alpha=0.7)
        ax_pid_components.plot(self.roll_data['time'], self.roll_data['I'], 'g-', label='I', alpha=0.7)
        ax_pid_components.plot(self.roll_data['time'], self.roll_data['D'], 'b-', label='D', alpha=0.7)
        if np.any(self.roll_data['FF']):  # Only plot if FF terms exist
            ax_pid_components.plot(self.roll_data['time'], self.roll_data['FF'], 'y-', label='FF', alpha=0.7)
        ax_pid_components.set_title('Roll PID Components')
        ax_pid_components.set_xlabel('Time (s)')
        ax_pid_components.set_ylabel('Output')
        ax_pid_components.legend()
        ax_pid_components.grid(True)
        
        # Create navigation data figure
        fig_nav, (ax_speed, ax_alt) = plt.subplots(2, 1, figsize=(15, 10))
        fig_nav.suptitle('Navigation Performance')
        
        # Plot airspeed data
        ax_speed.plot(self.nav_data['time'], self.nav_data['aspd'], 'b-', label='Airspeed', alpha=0.7)
        ax_speed.plot(self.nav_data['time'], self.nav_data['gspd'], 'r-', label='Ground Speed', alpha=0.7)
        if np.any(self.nav_data['target_aspd']):
            ax_speed.plot(self.nav_data['time'], self.nav_data['target_aspd'], 'g--', label='Target Airspeed', alpha=0.7)
        ax_speed.set_title('Speed')
        ax_speed.set_xlabel('Time (s)')
        ax_speed.set_ylabel('Speed (m/s)')
        ax_speed.legend()
        ax_speed.grid(True)
        
        # Save plots
        fig_pid.savefig('pid_analysis.png', bbox_inches='tight', dpi=300)
        fig_nav.savefig('nav_analysis.png', bbox_inches='tight', dpi=300)
        plt.close('all')
        
    def run_analysis(self, plot=False, verbose=False):
        """Run the complete analysis"""
        # Analyze each axis
        roll_analysis = self.analyze_pid_performance(self.roll_data, 'Roll')
        pitch_analysis = self.analyze_pid_performance(self.pitch_data, 'Pitch')
        yaw_analysis = self.analyze_pid_performance(self.yaw_data, 'Yaw')
        
        if verbose:
            print("\n=== PID Analysis Results ===\n")
            self._print_axis_analysis("Roll", roll_analysis)
            self._print_axis_analysis("Pitch", pitch_analysis)
            self._print_axis_analysis("Yaw", yaw_analysis)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(roll_analysis, pitch_analysis, yaw_analysis)
        
        # Print recommendations (removed from here since it's handled in main())
        
        if plot:
            self.plot_pid_data()
            print("\nPlots saved as pid_analysis.png and nav_analysis.png")
            
        return recommendations

    def _print_axis_analysis(self, axis_name, analysis):
        """Print detailed analysis results for a specific axis"""
        if not analysis['has_data']:
            print(f"{axis_name}: {analysis['message']}")
            return

        print(f"\n{axis_name} Axis Analysis:")
        print(f"  Oscillation Frequency: {analysis['oscillation_frequency']:.1f} Hz")
        print(f"  Mean Error: {analysis['mean_error']:.2f} deg")
        print(f"  Max Error: {analysis['max_error']:.2f} deg")
        print(f"  Steady State Error: {analysis['steady_state_error_pct']:.1f}%")
        print(f"  Steady State Bias: {analysis['steady_state_bias']:.2f} deg")
        print(f"  Overshoot: {analysis['overshoot_pct']:.1f}%")
        
        print("\n  PID Component Contributions:")
        print(f"    P: {analysis['p_contribution']:.1f}%")
        print(f"    I: {analysis['i_contribution']:.1f}%")
        print(f"    D: {analysis['d_contribution']:.1f}%")
        print(f"    FF: {analysis['ff_contribution']:.1f}%")
        
        print("\n  Average Term Values:")
        print(f"    P: {analysis['p_term_average']:.3f}")
        print(f"    I: {analysis['i_term_average']:.3f}")
        print(f"    D: {analysis['d_term_average']:.3f}")
        
        if analysis['dmod_max'] > analysis['dmod_min']:
            print(f"\n  D-term Modulation Range: {analysis['dmod_min']:.2f} - {analysis['dmod_max']:.2f}")
        
        print("\n  Oscillation Detection:")
        print(f"    P-term oscillation: {'Yes' if analysis['p_oscillation'] else 'No'}")
        print(f"    I-term oscillation: {'Yes' if analysis['i_oscillation'] else 'No'}")
        print(f"    D-term oscillation: {'Yes' if analysis['d_oscillation'] else 'No'}")

    def print_recommendations(self, recommendations):
        """Print PID tuning recommendations in a consistent format"""
        if not recommendations:
            print("No PID tuning recommendations")
            return

        print("\n=== PID Tuning Recommendations ===\n")
        
        # Group recommendations by controller
        controllers = {}
        for rec in recommendations:
            controller = 'RLL' if rec['param'].startswith('RLL') else 'PTCH' if rec['param'].startswith('PTCH') else 'YAW'
            if controller not in controllers:
                controllers[controller] = []
            controllers[controller].append(rec)

        # Sort controllers (RLL first, then PTCH, then YAW)
        controller_order = ['RLL', 'PTCH', 'YAW']
        
        for controller in controller_order:
            if controller in controllers:
                print(f"{controller} Controller:")
                # Sort parameters (P first, then I, then D)
                param_order = {'_P': 0, '_I': 1, '_D': 2}
                sorted_recs = sorted(controllers[controller], 
                                   key=lambda x: param_order.get(x['param'][-2:], 99))
                
                for rec in sorted_recs:
                    print(f"  {rec['param']}: Change from {rec['current']:.3f} to {rec['suggested']:.3f} - {rec['reason']}")
                print()

def main():
    args = parse_args()
    
    try:
        analyzer = PIDAnalyzer(args.logfile)
        analyzer.load_data()
        recommendations = analyzer.run_analysis(args.plot, args.verbose)
        
        # Print recommendations only once here
        analyzer.print_recommendations(recommendations)
        
        # Write recommendations to file if requested
        if args.output and recommendations:
            with open(args.output, 'w') as f:
                f.write("# PID Tuning Recommendations\n")
                f.write("# Generated by ArduPlane PID Analyzer\n\n")
                
                for controller, recs in controllers.items():
                    f.write(f"\n# {controller} Controller\n")
                    for rec in recs:
                        f.write(f"{rec['param']}={rec['suggested']:.3f}  # Changed from {rec['current']:.3f} - {rec['reason']}\n")
                        
            print(f"\nRecommendations written to {args.output}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
