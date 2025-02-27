import sys
import os
import shutil
import traceback
from tqdm import tqdm
from pymavlink import mavutil
from pymavlink import DFReader

def safe_path_create(directory):
    """
    Safely create directory, handling potential encoding or special character issues
    """
    try:
        # Normalize the path to handle potential encoding issues
        directory = os.path.normpath(directory)
        
        # Create directory with full permissions
        os.makedirs(directory, exist_ok=True)
        
        # Verify directory exists and is writable
        if not os.path.isdir(directory):
            raise ValueError(f"Failed to create directory: {directory}")
        
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"No write permission for directory: {directory}")
        
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        print(traceback.format_exc())
        return False

def zero_location_data(input_log_file):
    """
    Read an ArduPilot log file and zero out location data for specific message types
    
    Args:
        input_log_file (str): Path to the input log file
    
    Returns:
        str: Path to the output log file
    """
    try:
        # Resolve and validate input file path
        input_log_file = os.path.abspath(input_log_file)
        
        # Detailed input file validation
        if not input_log_file:
            print("Error: Empty input file path provided.")
            return None
        
        if not os.path.exists(input_log_file):
            print(f"Error: Input file does not exist: {input_log_file}")
            return None
        
        if not os.path.isfile(input_log_file):
            print(f"Error: Input path is not a file: {input_log_file}")
            return None
        
        # Generate output filename automatically
        file_dir = os.path.dirname(input_log_file)
        file_name = os.path.basename(input_log_file)
        file_name_without_ext, file_ext = os.path.splitext(file_name)
        output_log_file = os.path.join(file_dir, f"{file_name_without_ext}_no_loc{file_ext}")
        
        # Ensure output directory exists and is writable
        if not safe_path_create(file_dir):
            print(f"Failed to create or access directory: {file_dir}")
            return None
        
        # Copy the input file to output file first
        try:
            shutil.copy2(input_log_file, output_log_file)
        except Exception as copy_error:
            print(f"Error copying input file: {copy_error}")
            print(traceback.format_exc())
            return None
        
        # Specific message types to process
        target_msg_types = [
            'ADSB', 'AHR2', 'AIS1', 'AIS4', 'CAM', 'CMD', 'DSTL',
            'EAHR', 'FNCE', 'GPS', 'MISE', 'OAVG', 'ORGN', 'POS',
            'RALY', 'RGPJ', 'RSLL', 'RSO2', 'RSO3', 'SIM', 'TERR', 'TRIG',
            'NTUN'  # Added NTUN to the target message types
        ]
        
        # Comprehensive location-related field patterns to zero out
        location_field_patterns = [
            # Variations of latitude and longitude
            'lat', 'lon', 'long', 'latitude', 'longitude', 
            'Lat', 'Lon', 'Long', 'Latitude', 'Longitude',
            
            # Specific variations and prefixes
            'Lng', 'Lat', 
            'GPS_lat', 'GPS_lon', 'GPS_Lat', 'GPS_Lng',
            'gps_lat', 'gps_lon', 'gps_Lat', 'gps_Lng',
            
            # Potential compound or abbreviated forms
            'Latitude_deg', 'Longitude_deg',
            'lat_deg', 'lon_deg',
            'lat_rad', 'lon_rad',
            
            # Added specific NTUN fields
            'TLat', 'TLng'
        ]
        
        try:
            # Open the input log file
            log = DFReader.DFReader_binary(input_log_file)
            
            # Determine total number of messages for progress bar
            log.rewind()
            total_messages = sum(1 for _ in iter(log.recv_msg, None))
            log.rewind()
            
            # Open the output log file for writing
            try:
                output_log = open(output_log_file, 'wb')
            except Exception as write_error:
                print(f"Error creating output log file: {write_error}")
                print(f"Attempted path: {output_log_file}")
                print(traceback.format_exc())
                return None
            
            # Counters for tracking
            modified_msgs = 0
            
            # Progress bar with tqdm
            with tqdm(total=total_messages, desc="Processing log file", 
                      unit="msg", dynamic_ncols=True) as pbar:
                while True:
                    try:
                        # Read the next message
                        msg = log.recv_msg()
                        
                        if msg is None:
                            break
                        
                        # Check if the message type is in our target list
                        if msg.get_type() in target_msg_types:
                            # Dynamically find and zero out location fields
                            modified = False
                            for field in list(msg._fieldnames):  # Use list to avoid runtime modification
                                for pattern in location_field_patterns:
                                    # More aggressive matching
                                    if pattern.lower() in field.lower():
                                        try:
                                            # Set the field to 0
                                            setattr(msg, field, 0)
                                            modified = True
                                            # print(f"Zeroed field: {field} in {msg.get_type()}")
                                        except Exception as field_error:
                                            print(f"Could not zero field {field}: {field_error}")
                            
                            if modified:
                                modified_msgs += 1
                        
                        # Write the message to the output log
                        output_log.write(msg.get_msgbuf())
                        
                        # Update progress bar
                        pbar.update(1)
                    
                    except Exception as msg_error:
                        print(f"Error processing message: {msg_error}")
                        pbar.update(1)
                        continue
            
            # Close the output log file
            output_log.close()
            
            print(f"\nLocation data processing complete:")
            print(f"  Input file:  {input_log_file}")
            print(f"  Output file: {output_log_file}")
            print(f"  Total messages:     {total_messages}")
            print(f"  Modified messages:  {modified_msgs}")
            
            return output_log_file
        
        except Exception as processing_error:
            print(f"Error processing log file: {processing_error}")
            print(traceback.format_exc())
            return None
    
    except Exception as overall_error:
        print(f"Unexpected error: {overall_error}")
        print(traceback.format_exc())
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python strip_gps_data.py <input_log_file>")
        sys.exit(1)
    
    input_log_file = sys.argv[1]
    
    # Automatically generate and process output file
    zero_location_data(input_log_file)

if __name__ == "__main__":
    main()
    