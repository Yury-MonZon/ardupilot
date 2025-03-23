#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
from scipy.signal import find_peaks

def detect_step_input(setpoint_values, threshold=1.0, min_magnitude=15.0):
    """
    Detect step inputs in a time series of setpoint values
    
    Args:
        setpoint_values (np.array): Array of setpoint values
        threshold (float): Minimum step change to consider
        min_magnitude (float): Minimum absolute magnitude of step input
    
    Returns:
        list: Indices of step input start points
    """
    # Calculate differences between consecutive setpoint values
    diff_values = np.diff(setpoint_values)
    
    # Find indices where the absolute difference exceeds the threshold
    step_indices = np.where(np.abs(diff_values) > threshold)[0]
    
    # Filter step inputs by magnitude
    filtered_step_indices = [
        idx for idx in step_indices 
        if np.abs(setpoint_values[idx + 1] - setpoint_values[idx]) >= min_magnitude
    ]
    
    return filtered_step_indices

def analyze_pid_step_response(bin_file, output_dir=None, threshold=1.0, min_magnitude=15.0, max_steps=10):
    """
    Analyze PID step response from ArduPilot binary log
    
    Args:
        bin_file (str): Path to the ArduPilot binary log file
        output_dir (str, optional): Directory to save output graphs
        threshold (float, optional): Minimum step change to consider
        min_magnitude (float, optional): Minimum absolute magnitude of step input
        max_steps (int, optional): Maximum number of step inputs to analyze
    """
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(bin_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load the binary log
    mlog = mavutil.mavlink_connection(bin_file)

    # Data storage
    roll_data = {
        'setpoint': [],
        'measured': [],
        'timestamp': [],
        'mode': []
    }
    pitch_data = {
        'setpoint': [],
        'measured': [],
        'timestamp': [],
        'mode': []
    }

    # Detailed message type tracking
    message_types = {}
    mode_details = {}

    # Process log messages
    while True:
        try:
            m = mlog.recv_msg()
            if m is None:
                break

            # Track message types
            msg_type = m.get_type()
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

            # Detailed mode information
            if msg_type == 'MODE':
                # Print full details of MODE message
                mode_details[msg_type] = getattr(m, '__dict__', {})
                print(f"\nDEBUG: Full MODE Message Details:")
                for attr, value in vars(m).items():
                    print(f"  {attr}: {value}")

            # Detailed ATT message information
            if msg_type == 'ATT':
                # Print full details of ATT message
                print(f"\nDEBUG: Full ATT Message Details:")
                for attr, value in vars(m).items():
                    print(f"  {attr}: {value}")

        except Exception as e:
            print(f"Error processing message: {e}")
            break

    # Print message type summary
    print("\nDEBUG: Log Format Analysis")
    print("Message Type Occurrences:")
    for msg_type, count in sorted(message_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {msg_type}: {count}")

    # Attempt to print available attributes for MODE message
    print("\nAvailable Attributes for MODE Message:")
    try:
        sample_mode_msg = next(msg for msg in mlog.recv_msg() if msg.get_type() == 'MODE')
        print(dir(sample_mode_msg))
    except Exception as e:
        print(f"Could not retrieve MODE message attributes: {e}")

    # Reset log file to beginning
    mlog = mavutil.mavlink_connection(bin_file)

    # Current mode tracking
    current_mode = None
    current_mode_num = None
    mode_counts = {}

    # Process log messages again for data extraction
    while True:
        try:
            m = mlog.recv_msg()
            if m is None:
                break

            # Extract mode information first
            if m.get_type() == 'MODE':
                current_mode = getattr(m, 'Mode', None)
                current_mode_num = getattr(m, 'ModeNum', None)
                
                # Count mode occurrences
                mode_counts[current_mode] = mode_counts.get(current_mode, 0) + 1
                continue

            # Extract ATT messages
            if m.get_type() == 'ATT':
                roll_data['setpoint'].append(m.DesRoll)
                roll_data['measured'].append(m.Roll)
                roll_data['timestamp'].append(m.TimeUS / 1e6)  # Convert to seconds
                
                # Use mode number for cruise mode detection
                is_cruise_mode = (
                    current_mode_num is not None and 
                    current_mode_num in [10, 11]  # Guided and Cruise modes
                )
                roll_data['mode'].append(is_cruise_mode)

                pitch_data['setpoint'].append(m.DesPitch)
                pitch_data['measured'].append(m.Pitch)
                pitch_data['timestamp'].append(m.TimeUS / 1e6)  # Convert to seconds
                pitch_data['mode'].append(is_cruise_mode)

        except Exception as e:
            print(f"Error processing message: {e}")
            break

    # Print mode detection summary
    print("\nDEBUG: Mode Detection Summary")
    print("Total Mode Occurrences:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")
    
    print("\nRoll Mode Data:")
    print(f"  Total data points: {len(roll_data['mode'])}")
    print(f"  Cruise mode points: {sum(roll_data['mode'])}")
    
    print("\nPitch Mode Data:")
    print(f"  Total data points: {len(pitch_data['mode'])}")
    print(f"  Cruise mode points: {sum(pitch_data['mode'])}")

    # Convert to numpy arrays
    roll_data = {k: np.array(v) for k, v in roll_data.items()}
    pitch_data = {k: np.array(v) for k, v in pitch_data.items()}

    # Analyze step responses
    def analyze_step_response(data, axis_name):
        # Detect step inputs
        step_indices = detect_step_input(data['setpoint'], threshold, min_magnitude)
        
        if len(step_indices) == 0:
            print(f"No significant step inputs found for {axis_name}")
            return None

        # Filter step inputs in cruise mode
        cruise_step_indices = [
            idx for idx in step_indices 
            if len(data['mode']) > idx and data['mode'][idx]
        ]
        
        if len(cruise_step_indices) == 0:
            print(f"No step inputs found in cruise mode for {axis_name}")
            return None

        # Limit number of step inputs
        cruise_step_indices = cruise_step_indices[:max_steps]

        # Analyze each step input
        step_results = []
        for step_index in cruise_step_indices:
            # Extract response around the step
            pre_step = data['measured'][:step_index]
            post_step = data['measured'][step_index:]
            time_pre = data['timestamp'][:step_index]
            time_post = data['timestamp'][step_index:]
            
            # Calculate key metrics
            settling_threshold = 0.02  # 2% of final value
            final_value = post_step[-1]
            
            # Rise time (10% to 90% of final value)
            rise_indices = np.where((post_step >= 0.1 * final_value) & (post_step <= 0.9 * final_value))[0]
            rise_time = time_post[rise_indices[-1]] - time_post[rise_indices[0]] if len(rise_indices) > 0 else None
            
            # Settling time
            settling_indices = np.where(np.abs(post_step - final_value) <= settling_threshold * final_value)[0]
            settling_time = time_post[settling_indices[0]] if len(settling_indices) > 0 else None
            
            # Overshoot
            peak_indices, _ = find_peaks(post_step)
            max_overshoot = max(post_step[peak_indices]) if len(peak_indices) > 0 else final_value
            overshoot_percent = ((max_overshoot - final_value) / final_value) * 100 if max_overshoot > final_value else 0
            
            # Plot step response
            plt.figure(figsize=(10, 6))
            plt.plot(time_pre, pre_step, label='Pre-Step', color='blue')
            plt.plot(time_post, post_step, label='Post-Step', color='red')
            plt.title(f'{axis_name} PID Step Response (Step {len(step_results) + 1})')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (degrees)')
            plt.legend()
            plt.grid(True)
            
            # Annotate metrics
            plt.annotate(f'Rise Time: {rise_time:.3f}s' if rise_time else 'Rise Time: N/A', 
                         xy=(0.05, 0.95), xycoords='axes fraction')
            plt.annotate(f'Settling Time: {settling_time:.3f}s' if settling_time else 'Settling Time: N/A', 
                         xy=(0.05, 0.90), xycoords='axes fraction')
            plt.annotate(f'Overshoot: {overshoot_percent:.2f}%', xy=(0.05, 0.85), xycoords='axes fraction')
            
            # Save plot
            output_file = os.path.join(output_dir, f'{axis_name.lower()}_step_response_{len(step_results) + 1}.png')
            plt.savefig(output_file)
            plt.close()
            
            # Store results
            step_results.append({
                'rise_time': rise_time,
                'settling_time': settling_time,
                'overshoot_percent': overshoot_percent
            })
        
        return step_results

    # Analyze roll and pitch
    roll_metrics = analyze_step_response(roll_data, 'Roll')
    pitch_metrics = analyze_step_response(pitch_data, 'Pitch')

    # Print metrics
    print("\nPID Step Response Metrics:")
    
    def print_axis_metrics(axis_name, metrics):
        if not metrics:
            print(f"\n{axis_name} Axis: No step responses found")
            return
        
        # Calculate average metrics
        avg_metrics = {
            'rise_time': np.mean([m['rise_time'] for m in metrics if m['rise_time'] is not None]),
            'settling_time': np.mean([m['settling_time'] for m in metrics if m['settling_time'] is not None]),
            'overshoot_percent': np.mean([m['overshoot_percent'] for m in metrics])
        }
        
        print(f"\n{axis_name} Axis (Average of {len(metrics)} Step Inputs):")
        print(f"  Rise Time: {avg_metrics['rise_time']:.3f}s")
        print(f"  Settling Time: {avg_metrics['settling_time']:.3f}s")
        print(f"  Overshoot: {avg_metrics['overshoot_percent']:.2f}%")

    print_axis_metrics('Roll', roll_metrics)
    print_axis_metrics('Pitch', pitch_metrics)

def main():
    parser = argparse.ArgumentParser(description='Analyze PID Step Response from ArduPilot Log')
    parser.add_argument('input_log', help='Input ArduPilot binary log file')
    parser.add_argument('--output-dir', help='Directory to save output graphs', default=None)
    parser.add_argument('--threshold', type=float, default=1.0, 
                        help='Minimum step change to consider (default: 1.0 degrees)')
    parser.add_argument('--min-magnitude', type=float, default=15.0, 
                        help='Minimum absolute magnitude of step input (default: 15.0 degrees)')
    parser.add_argument('--max-steps', type=int, default=10, 
                        help='Maximum number of step inputs to analyze (default: 10)')
    
    args = parser.parse_args()
    
    analyze_pid_step_response(
        args.input_log, 
        args.output_dir, 
        args.threshold, 
        args.min_magnitude, 
        args.max_steps
    )

if __name__ == "__main__":
    main()