#!/usr/bin/env python3
"""
Convert ArduPilot binary logs to BBL format for PID Analyzer
"""

import sys
import os
import argparse
import numpy as np
from pymavlink import mavutil
from tqdm.auto import tqdm
import uuid
from datetime import datetime
import shlex

def extract_pid_data(bin_file, output_file=None, max_messages=None):
    """
    Extract PID data from ArduPilot binary log and convert to BBL format for Plasmatree PID Analyzer
    """
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(bin_file))[0]
        output_file = os.path.join(os.path.dirname(bin_file), f"{base_name}.bbl")

    mlog = mavutil.mavlink_connection(bin_file)
    
    # Initialize data structures for all axes
    axes_data = {
        'roll': [],
        'pitch': [],
        'yaw': [],
        'throttle': []  # Added throttle axis
    }
    
    msg_count = 0
    start_time = None
    
    print("Reading log messages...")
    
    # First pass - count messages for progress bar
    msg_types = ['PIDR', 'PIDP', 'PIDY']
    total_messages = 0
    while True:
        msg = mlog.recv_match(type=msg_types, blocking=False)
        if msg is None:
            break
        total_messages += 1
    
    # Reset log reading
    mlog = mavutil.mavlink_connection(bin_file)
    
    # Second pass - actual processing
    with tqdm(total=total_messages, desc="Processing messages") as pbar:
        while True:
            msg = mlog.recv_match(type=msg_types, blocking=False)
            if msg is None:
                break
                
            if start_time is None:
                start_time = msg.TimeUS
                
            msg_type = msg.get_type()
            timestamp = msg.TimeUS - start_time  # Relative timestamp in microseconds
            
            # Map message type to axis
            axis_map = {'PIDR': 'roll', 'PIDP': 'pitch', 'PIDY': 'yaw'}
            axis = axis_map[msg_type]
            
            axes_data[axis].append({
                'timestamp': timestamp,
                'target': msg.Tar,
                'actual': msg.Act,
                'P': msg.P,
                'I': msg.I,
                'D': msg.D,
                'FF': msg.FF if hasattr(msg, 'FF') else 0
            })
            
            msg_count += 1
            pbar.update(1)
                
            if max_messages and msg_count >= max_messages:
                break

    if not any(axes_data.values()):
        raise ValueError("No PID data found in log file")
    
    print("\nProcessing data...")
    
    # Sort data for each axis
    for axis_data in axes_data.values():
        if axis_data:
            axis_data.sort(key=lambda x: x['timestamp'])
    
    try:
        print("Writing BBL file...")
        with open(output_file, 'w') as f:
            # Write headers
            f.write("H Product:Blackbox flight data recorder by Nicholas Sherlock\n")
            f.write("H Data version:2\n")
            f.write("H Firmware type:Betaflight\n")
            f.write("H Firmware revision:4.3\n")
            
            # Required metadata
            f.write("H P interval:16666\n")
            f.write("H minthrottle:1000\n")
            f.write("H maxthrottle:2000\n")
            f.write("H gyro_scale:16.4\n")
            f.write("H motorOutput:0,1000,2000\n")
            f.write("H vbatscale:110\n")
            f.write("H vbatref:420\n")
            f.write("H vbatcellvoltage:330,350,430\n")
            f.write("H acc_1G:4096\n")
            f.write("H looptime:125\n")
            
            # Field definitions for all axes (including throttle)
            for i in range(4):  # 0=roll, 1=pitch, 2=yaw, 3=throttle
                # RC Command
                f.write(f"H Field P{i} name:rcCommand[{i}]\n")
                f.write(f"H Field P{i} signed:1\n")
                f.write(f"H Field P{i} encoding:SIGNED_VB\n")
                f.write(f"H Field P{i} predictor:0\n")
                
                # Gyro
                f.write(f"H Field G{i} name:gyroADC[{i}]\n")
                f.write(f"H Field G{i} signed:1\n")
                f.write(f"H Field G{i} encoding:SIGNED_VB\n")
                f.write(f"H Field G{i} predictor:0\n")
                
                # D term
                f.write(f"H Field D{i} name:axisD[{i}]\n")
                f.write(f"H Field D{i} signed:1\n")
                f.write(f"H Field D{i} encoding:SIGNED_VB\n")
                f.write(f"H Field D{i} predictor:0\n")
                
                # PID Sum
                f.write(f"H Field S{i} name:axisSum[{i}]\n")
                f.write(f"H Field S{i} signed:1\n")
                f.write(f"H Field S{i} encoding:SIGNED_VB\n")
                f.write(f"H Field S{i} predictor:0\n")
            
            # Time field
            f.write("H Field I name:time\n")
            f.write("H Field I signed:0\n")
            f.write("H Field I encoding:UNSIGNED_VB\n")
            f.write("H Field I predictor:0\n")
            
            # Get earliest timestamp across all axes
            min_timestamp = float('inf')
            for axis_data in axes_data.values():
                if axis_data:
                    min_timestamp = min(min_timestamp, axis_data[0]['timestamp'])
            
            # Write initial timestamp
            f.write(f"E {min_timestamp}\n")
            
            # Combine and sort all data points
            all_timestamps = set()
            for axis_data in axes_data.values():
                all_timestamps.update(entry['timestamp'] for entry in axis_data)
            
            all_timestamps = sorted(all_timestamps)
            last_timestamp = None
            
            print("Writing data frames...")
            with tqdm(total=len(all_timestamps), desc="Writing frames") as pbar:
                for timestamp in all_timestamps:
                    # Find data for each axis at this timestamp
                    frame_data = [str(int((timestamp - min_timestamp) / 1000))]  # Time in ms
                    
                    # Process all axes including throttle
                    for i in range(4):
                        axis = ['roll', 'pitch', 'yaw', 'throttle'][i]
                        matching_data = next(
                            (entry for entry in axes_data[axis] if entry['timestamp'] == timestamp),
                            {'target': 0, 'actual': 0, 'P': 0, 'I': 0, 'D': 0}
                        )
                        
                        frame_data.extend([
                            str(int(matching_data['target'] * 500)),  # RC Command
                            str(int(matching_data['actual'] * 500)),  # Gyro
                            str(int(matching_data['D'] * 1000)),  # D term
                            str(int((matching_data['P'] + matching_data['I'] + matching_data['D']) * 1000))  # Axis sum
                        ])
                    
                    f.write("I " + ",".join(frame_data) + "\n")
                    
                    # Write event frame for significant time changes
                    if last_timestamp is None or (timestamp - last_timestamp) > 16666:
                        f.write(f"E {timestamp}\n")
                        last_timestamp = timestamp
                    
                    pbar.update(1)
            
        print(f"\nConverted {msg_count} messages")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing log: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert ArduPilot bin log to PID Analyzer BBL format')
    parser.add_argument('input_log', help='Input binary log file')
    parser.add_argument('output_file', nargs='?', help='Optional output BBL file (will auto-generate if not provided)')
    parser.add_argument('--max-messages', type=int, default=None, 
                      help='Maximum number of messages to process')
    
    args = parser.parse_args()
    
    # Handle paths with spaces
    input_log = os.path.expanduser(args.input_log)
    output_file = os.path.expanduser(args.output_file) if args.output_file else None
    
    # Validate input file exists
    if not os.path.exists(input_log):
        print(f"Error: Input file not found: {input_log}")
        sys.exit(1)
        
    extract_pid_data(input_log, output_file, args.max_messages)

if __name__ == "__main__":
    main()
