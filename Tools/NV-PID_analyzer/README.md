# NV-PID Analyzer

ArduPlane PID/FF Parameter Analysis Tool

## Overview

NV-PID Analyzer analyzes ArduPlane binary logs to evaluate PIDFF tuning and provide tuning suggestions based on ArduPilot autotune logic.

## Features

- **Roll and Pitch rate controller analysis**
- **Step response metrics**: Rise time, overshoot, settling time, steady-state error
- **Oscillation detection** via Dmod analysis
- **Tuning suggestions** based on ArduPilot autotune logic
- **Autotune session analysis** from ATRP messages
- **Flight mode awareness**: Analyzes post-autotune flight modes
- **Robust noise handling**: Uses convolution methods for real flight data
- **Adaptive sample rate detection**: Works with any log sample rate

## Requirements

```bash
pip install pymavlink numpy matplotlib scipy
```

Or use the ArduPilot virtual environment:
```bash
source /path/to/venv-ardupilot/bin/activate
```

## Usage

### Basic Analysis

```bash
python nv_pid_analyzer.py <logfile.bin>
```

### Options

```
--axis {roll,pitch,both}    Axis to analyze (default: both)
--output, -o DIR            Output directory for plots (default: ./plots)
--no-plots                  Skip generating plots
```

### Examples

```bash
# Analyze both roll and pitch
python nv_pid_analyzer.py log1.bin

# Analyze only roll
python nv_pid_analyzer.py log1.bin --axis roll

# Custom output directory
python nv_pid_analyzer.py log1.bin --output ./my_plots
```

## Output

### Console Output

The tool provides:

1. **Parameters**: Current PID values and AUTOTUNE_LEVEL
2. **Flight Analysis**: Autotune sessions and analysis phases detected
3. **Roll/Pitch Analysis**:
   - Tracking performance (RMS error, correlation)
   - Oscillation check (Dmod analysis)
   - Step response metrics
   - PID component contributions
   - Overall quality score
4. **Tuning Suggestions**: Prioritized recommendations
5. **Summary**: Critical issues count

### Generated Plots

- `roll_step_response.png` / `pitch_step_response.png`: Step response analysis
- `roll_pid_components.png` / `pitch_pid_components.png`: PID components over time
- `autotune_progression.png`: Autotune gain progression
- `roll_error_distribution.png` / `pitch_error_distribution.png`: Error histograms
- `roll_tracking_scatter.png` / `pitch_tracking_scatter.png`: Target vs actual scatter

## Key Concepts

### Dmod (Oscillation Detection)

- `Dmod = 1.0`: No oscillation
- `Dmod < 1.0`: Oscillation detected (P+D slew rate limiting)
- `Dmod < 0.9`: Minor oscillation
- `Dmod < 0.7`: Moderate oscillation  
- `Dmod < 0.5`: Severe oscillation

### AUTOTUNE_LEVEL Targets

| Level | Tau (s) | Rmax (deg/s) | Max Overshoot | Max Rise Time |
|-------|---------|--------------|---------------|---------------|
| 1     | 1.00    | 20           | 20%           | 1.0s          |
| 5     | 0.60    | 60           | 15%           | 0.6s          |
| 7     | 0.30    | 90           | 12%           | 0.4s          |
| 10    | 0.10    | 210          | 8%            | 0.2s          |

### Tuning Rules

Based on ArduPilot autotune logic:

1. **Oscillation (Dmod < 1.0)** → Decrease D gain
2. **High overshoot (>target)** → Decrease P gain
3. **Slow response (>target rise time)** → Increase P or FF
4. **Steady-state error (>1 deg/s)** → Increase I gain
5. **Poor tracking with low FF** → Increase FF gain

## Flight Mode Handling

The tool automatically:
- Detects autotune sessions from MSG messages
- Analyzes **post-autotune** flight data only
- Skips oscillation detection during autotune (oscillation is intentional)
- If no autotune exists, analyzes all suitable flight modes (CRUISE, FBWA, LOITER)

## File Structure

```
NV-PID_analyzer/
├── nv_pid_analyzer.py    # Main entry point
├── log_parser.py         # Binary log parsing
├── flight_analyzer.py    # Flight phase detection
├── step_detector.py      # Step detection and metrics
├── pid_metrics.py        # Tracking and oscillation analysis
├── tuning_suggester.py   # Tuning recommendations
├── visualizer.py         # Plot generation
└── README.md             # This file
```

## Algorithm Details

### Step Detection

1. Smooth target signal using Gaussian + Savitzky-Golay cascade
2. Calculate derivative of smoothed signal
3. Find peaks where derivative exceeds threshold (adaptive)
4. Validate steps by size and duration
5. Merge nearby steps

### Noise Handling

- Gaussian convolution smoothing (adaptive to sample rate)
- Savitzky-Golay filter for peak preservation
- Hysteresis thresholding for step detection
- Robust statistics (median instead of mean)

## License

Same as ArduPilot - GNU General Public License v3

## References

- [ArduPilot Autotune](https://ardupilot.org/plane/docs/autotune.html)
- [AP_AutoTune.cpp](https://github.com/ArduPilot/ardupilot/blob/master/libraries/APM_Control/AP_AutoTune.cpp)
- [AC_PID.cpp](https://github.com/ArduPilot/ardupilot/blob/master/libraries/AC_PID/AC_PID.cpp)
