# ArduPilot PIDFF Simulator

Educational simulator for tuning ArduPilot PIDFF (PID + Feed Forward) controllers for fixed-wing aircraft roll and pitch axes at cruise speed.

**Based on actual ArduPilot implementation:**
- `libraries/APM_Control/AP_RollController.h/cpp`
- `libraries/APM_Control/AP_PitchController.h/cpp`
- `libraries/AC_PID/AC_PID.h/cpp`

## Purpose

This tool helps users learn how to tune ArduPilot fixed-wing PID parameters by:
- Visualizing step responses in real-time
- Showing individual PIDFF component contributions
- Randomizing aircraft dynamics to practice retuning
- Providing autotune functionality
- Simulating airspeed scaling effects

## Features

- **Real-time graphs**: Step response and PIDFF component visualization
- **Step input testing**: +45°/-45° roll step with 2-second hold
- **ArduPilot parameters**: Uses actual APM_Control naming (`SRV2SRV_RATE_*`, `SRV1SRV_RATE_*`)
- **Airspeed scaling**: Simulates control authority changes with airspeed
- **Aircraft randomization**: Randomly changes aircraft dynamics to simulate different airframes
- **Autotune**: Relay-based autotuning algorithm (matches ArduPilot AP_AutoTune)
- **GTK/GNOME native**: Uses PyGObject, not Qt

## Installation

Requires Python 3.10+ and UV package manager.

```bash
# Install dependencies with UV
uv sync

# Run the simulator
uv run python pidff_simulator.py
```

## Usage

### Basic Operation

1. Click **▶ Start** to begin the simulation
2. Click **+45° Roll Step** or **-45° Roll Step** to initiate a step test
3. Observe the response in the graphs
4. Adjust PID parameters using the spinboxes
5. Repeat until you achieve good response

### ArduPilot Parameters

The simulator uses the same parameter names as ArduPilot's APM_Control library:

| Parameter | Description | Default (Roll) | Default (Pitch) | Range |
|-----------|-------------|----------------|-----------------|-------|
| `_RATE_P` | Rate proportional gain | 0.08 | 0.04 | 0.01-0.5 |
| `_RATE_I` | Rate integral gain | 0.15 | 0.15 | 0.01-0.5 |
| `_RATE_D` | Rate derivative gain | 0.0 | 0.0 | 0.0-0.05 |
| `_RATE_FF` | Feed forward gain | 0.345 | 0.345 | 0.1-1.0 |
| `_RATE_IMAX` | Integral limit | 0.666 | 0.666 | 0.1-1.0 |
| `_RATE_FLTT` | Target filter (Hz) | 3 | 3 | 1-20 |

In ArduPilot Mission Planner, these appear as:
- Roll: `SRV2SRV_RATE_P`, `SRV2SRV_RATE_I`, etc.
- Pitch: `SRV1SRV_RATE_P`, `SRV1SRV_RATE_I`, etc.

### Tuning Guidelines

**Good tuning characteristics:**
- Fast rise time with minimal overshoot (<10%)
- Quick settling time
- Zero steady-state error
- FF provides most of the control, P corrects errors

**Tuning order:**
1. Start with default FF (~0.35)
2. Increase P until slight overshoot, then reduce slightly
3. Add I if there's steady-state error (usually minimal needed)
4. Add D only if there's oscillation (often not needed)
5. Adjust FF to improve tracking

**Parameter effects:**
- **_RATE_P**: Proportional gain - affects response speed and overshoot
- **_RATE_I**: Integral gain - eliminates steady-state error
- **_RATE_D**: Derivative gain - damping, reduces overshoot (use sparingly)
- **_RATE_FF**: Feed forward - primary control action, improves tracking

### Scaler (Airspeed Scaling)

ArduPilot scales PID gains based on airspeed:
- `scaler = scaling_speed / actual_airspeed`
- When flying slower than tuning speed, controls are more effective
- When flying faster, controls are less effective

The simulator shows the current scaler value. At cruise (scaling_speed = actual_airspeed), scaler = 1.0.

### Randomize Mode

Click **🎲 Randomize Aircraft** to change the simulated aircraft dynamics. This simulates different airframes and requires you to retune the PIDs - just like tuning a real aircraft!

Variations include:
- Roll/pitch damping changes
- Control power changes
- Stability changes

### Autotune

Click **🔧 Autotune** to let the simulator automatically tune the PIDFF parameters using a relay feedback method (similar to ArduPilot's AP_AutoTune).

## Default Parameters

Typical ArduPlane values for a trainer aircraft at cruise:

| Parameter | Roll | Pitch |
|-----------|------|-------|
| _RATE_P | 0.08 | 0.04 |
| _RATE_I | 0.15 | 0.15 |
| _RATE_D | 0.0 | 0.0 |
| _RATE_FF | 0.345 | 0.345 |
| _RATE_IMAX | 0.666 | 0.666 |

## Technical Details

- **Simulation rate**: 50Hz (matching ArduPlane main loop)
- **Aircraft model**: 2nd-order dynamics with airspeed scaling
- **PIDFF structure**: Matches ArduPilot APM_Control exactly
  - Radians internally, degrees for I/O
  - Output in servo centidegrees (-4500 to 4500)
  - Target, error, and derivative filters
  - Airspeed scaling (scaler²)
- **GUI framework**: GTK 4 with libadwaita

## Requirements

- Python 3.10+
- PyGObject (GTK 4)
- NumPy
- libadwaita

## License

Educational tool for ArduPilot community.
