#!/usr/bin/env python3
"""
ArduPilot PIDFF Simulator for Roll Tuning Education
Uses ArduPilot fixed-wing parameters and scales (APM_Control library)
GNOME/GTK based GUI (no Qt)

Based on actual ArduPilot implementation in:
- libraries/APM_Control/AP_RollController.h/cpp
- libraries/AC_PID/AC_PID.h/cpp

Simulates roll axis only at cruise speed.
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Adw, GLib, Gdk

import numpy as np
from collections import deque
import random

# ============================================================================
# ArduPilot PIDFF Controller Implementation (matches APM_Control)
# ============================================================================

class ArduPilotPIDFF:
    """
    ArduPilot fixed-wing PIDFF controller matching APM_Control implementation.
    
    Parameters (matching ArduPilot naming convention):
    - _RATE_P: Rate proportional gain (default: 0.08 roll, 0.04 pitch)
    - _RATE_I: Rate integral gain (default: 0.15)
    - _RATE_D: Rate derivative gain (default: 0)
    - _RATE_FF: Feed forward gain (default: 0.345)
    - _RATE_IMAX: Integral limit (default: 0.666, scaled to servo output)
    - _RATE_FLTT: Target filter frequency in Hz (default: 3)
    - _RATE_FLTE: Error filter frequency in Hz (default: 12)
    - _RATE_FLTD: Derivative filter frequency in Hz (default: 150)
    - _RATE_SMAX: Slew rate limit (default: 0, disabled)
    - _RATE_PDMX: PD sum maximum (default: 1)
    - _RATE_D_FF: Derivative feedforward gain (default: 0)
    
    The controller runs in radians internally but accepts deg/s for input/output.
    Output is scaled to centidegrees of servo deflection (-4500 to 4500).
    """
    
    def __init__(self, dt=0.02, axis='roll'):
        self.dt = dt  # 50Hz main loop (typical ArduPlane rate)
        self.axis = axis
        
        # Default ArduPlane values from AC_PID constructor
        # AC_PID rate_pid{0.08, 0.15, 0, 0.345, 0.666, 3, 0, 12, 150, 1};
        if axis == 'roll':
            self.p_gain = 0.08      # _RATE_P (default roll)
        else:
            self.p_gain = 0.04      # _RATE_P (default pitch)
        self.i_gain = 0.15          # _RATE_I
        self.d_gain = 0.0           # _RATE_D
        self.ff_gain = 0.345        # _RATE_FF
        self.imax = 0.666           # _RATE_IMAX (normalized, scaled by 4500 for output)
        self.target_filter_hz = 3.0   # _RATE_FLTT
        self.error_filter_hz = 12.0   # _RATE_FLTE
        self.deriv_filter_hz = 150.0  # _RATE_FLTD
        self.slew_max = 0.0         # _RATE_SMAX (0 = disabled)
        self.pd_max = 1.0           # _RATE_PDMX
        self.d_ff_gain = 0.0        # _RATE_D_FF
        
        # Internal state (radians for internal calculations)
        self.integrator = 0.0       # I term (radians)
        self.filtered_target = 0.0  # Filtered target rate (rad/s)
        self.filtered_error = 0.0   # Filtered error (rad/s)
        self.prev_derivative = 0.0  # Previous derivative (rad/s²)
        self.prev_actual_rate = 0.0 # Previous actual rate for D term (rad/s)
        self.prev_target = 0.0      # Previous target for D-FF (rad/s)
        
        # Filter coefficients
        self._update_filters()
        
        # Last output components for display (deg/s equivalent)
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_ff = 0.0
        self.last_dff = 0.0
        self.last_output = 0.0
        
    def _update_filters(self):
        """Update filter coefficients based on frequencies."""
        # First-order lowpass: y[n] = a1*y[n-1] + b1*x[n]
        # where a1 = 1 - alpha, b1 = alpha, alpha = dt / (tau + dt)
        for name, hz in [('target', self.target_filter_hz),
                         ('error', self.error_filter_hz),
                         ('deriv', self.deriv_filter_hz)]:
            tau = 1.0 / (2 * np.pi * hz)
            alpha = self.dt / (tau + self.dt)
            setattr(self, f'{name}_filter_a1', 1.0 - alpha)
            setattr(self, f'{name}_filter_b1', alpha)
        
    def reset(self):
        """Reset controller state."""
        self.integrator = 0.0
        self.filtered_target = 0.0
        self.filtered_error = 0.0
        self.prev_derivative = 0.0
        self.prev_actual_rate = 0.0
        self.prev_target = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_ff = 0.0
        self.last_dff = 0.0
        self.last_output = 0.0
        
    def _lowpass(self, value, prev, a1, b1):
        """Apply first-order lowpass filter."""
        return a1 * prev + b1 * value
        
    def update(self, target_rate_deg, actual_rate_deg, scaler=1.0, disable_integrator=False, ground_mode=False):
        """
        Update PIDFF controller matching ArduPilot APM_Control implementation.
        
        Args:
            target_rate_deg: Desired angular rate (deg/s)
            actual_rate_deg: Measured angular rate (gyro, deg/s)
            scaler: Airspeed scaling factor (scaling_speed / actual_airspeed)
            disable_integrator: Disable I term (for ground mode or special cases)
            ground_mode: Suppress D term to prevent ground oscillations
            
        Returns:
            Control output in centidegrees of servo deflection (-4500 to 4500)
        """
        # Convert to radians for internal calculations (matching ArduPilot)
        target_rate_rad = np.radians(target_rate_deg)
        actual_rate_rad = np.radians(actual_rate_deg)
        
        # Apply target filter (matching ArduPilot _RATE_FLTT)
        self.filtered_target = self._lowpass(
            target_rate_rad, self.filtered_target,
            self.target_filter_a1, self.target_filter_b1
        )
        
        # Calculate error and apply error filter (matching _RATE_FLTE)
        error_rad = self.filtered_target - actual_rate_rad
        self.filtered_error = self._lowpass(
            error_rad, self.filtered_error,
            self.error_filter_a1, self.error_filter_b1
        )
        
        # P term (on filtered error)
        # Note: ArduPilot scales by scaler^2 for airspeed compensation
        p_out = self.filtered_error * self.p_gain * (scaler ** 2)
        
        # I term with anti-windup
        if not disable_integrator:
            self.integrator += self.filtered_error * self.i_gain * (scaler ** 2) * self.dt
            # Clamp integrator to IMAX (scaled by 4500 for servo output range)
            imax_rad = self.imax  # IMAX is in normalized units
            self.integrator = np.clip(self.integrator, -imax_rad, imax_rad)
        i_out = self.integrator
        
        # D term (on measurement - gyro feedback, matching ArduPilot)
        # Derivative of actual rate (negative for damping)
        raw_derivative = -(actual_rate_rad - self.prev_actual_rate) / self.dt
        # Apply derivative filter (matching _RATE_FLTD)
        filtered_derivative = self._lowpass(
            raw_derivative, self.prev_derivative,
            self.deriv_filter_a1, self.deriv_filter_b1
        )
        self.prev_derivative = filtered_derivative
        self.prev_actual_rate = actual_rate_rad  # Store for next derivative calculation
        d_out = filtered_derivative * self.d_gain
        
        # D-FF term (derivative of target, matching _RATE_D_FF)
        target_derivative = (target_rate_rad - self.prev_target) / self.dt
        self.prev_target = target_rate_rad
        dff_out = target_derivative * self.d_ff_gain
        
        # FF term (on target rate, matching _RATE_FF)
        # Note: ArduPilot divides by (scaler * eas2tas) for proper scaling
        # We simplify by using scaler only (eas2tas = 1 for simulation)
        # Positive target rate → positive servo output → positive roll rate
        ff_out = target_rate_rad * self.ff_gain / max(scaler, 0.1)
        
        # Sum components (in radians)
        out_rad = p_out + i_out + d_out + ff_out + dff_out
        
        # Apply PD max limit if configured
        if self.pd_max > 0:
            pd_sum = p_out + d_out
            if abs(pd_sum) > self.pd_max:
                scale = self.pd_max / abs(pd_sum)
                p_out *= scale
                d_out *= scale
                out_rad = p_out + i_out + d_out + ff_out + dff_out
        
        # Apply slew rate limit if configured
        if self.slew_max > 0:
            slew_limit = np.radians(self.slew_max) * self.dt
            out_rad = np.clip(out_rad, self.last_output/100.0 - slew_limit, self.last_output/100.0 + slew_limit)
        
        # Ground mode: suppress D term to prevent oscillations
        if ground_mode:
            out_rad -= d_out + 0.5 * p_out
        
        # Convert to servo output (centidegrees, matching ArduPilot -4500 to 4500)
        # Positive output = right stick = right wing down = positive roll rate
        output_cd = np.degrees(out_rad) * 100
        output_cd = np.clip(output_cd, -4500, 4500)
        
        # Store components for display (convert to deg/s equivalent for display)
        self.last_p = np.degrees(p_out) * 100
        self.last_i = np.degrees(i_out) * 100
        self.last_d = np.degrees(d_out) * 100
        self.last_ff = np.degrees(ff_out) * 100
        self.last_dff = np.degrees(dff_out) * 100
        self.last_output = output_cd
        
        return output_cd
    
    def get_params_dict(self):
        """Return parameters as dictionary with ArduPilot names."""
        return {
            '_RATE_P': self.p_gain,
            '_RATE_I': self.i_gain,
            '_RATE_D': self.d_gain,
            '_RATE_FF': self.ff_gain,
            '_RATE_IMAX': self.imax,
            '_RATE_FLTT': self.target_filter_hz,
            '_RATE_FLTE': self.error_filter_hz,
            '_RATE_FLTD': self.deriv_filter_hz,
            '_RATE_SMAX': self.slew_max,
            '_RATE_PDMX': self.pd_max,
            '_RATE_D_FF': self.d_ff_gain,
        }
    
    def set_from_dict(self, params):
        """Set parameters from dictionary."""
        for key, attr in [('_RATE_P', 'p_gain'), ('_RATE_I', 'i_gain'),
                          ('_RATE_D', 'd_gain'), ('_RATE_FF', 'ff_gain'),
                          ('_RATE_IMAX', 'imax'), ('_RATE_FLTT', 'target_filter_hz'),
                          ('_RATE_FLTE', 'error_filter_hz'), ('_RATE_FLTD', 'deriv_filter_hz'),
                          ('_RATE_SMAX', 'slew_max'), ('_RATE_PDMX', 'pd_max'),
                          ('_RATE_D_FF', 'd_ff_gain')]:
            if key in params:
                setattr(self, attr, params[key])
        self._update_filters()


# ============================================================================
# Aircraft Dynamics Model (Fixed-Wing at Cruise)
# ============================================================================

class AircraftDynamics:
    """
    Simplified fixed-wing aircraft dynamics model for roll and pitch at cruise speed.
    Matches ArduPilot's approach to aircraft response.
    
    Key characteristics:
    - Airspeed-dependent control authority (scaler = scaling_speed / actual_airspeed)
    - First-order actuator response
    - Natural damping and stability
    """
    
    def __init__(self, dt=0.02):
        self.dt = dt
        
        # Cruise flight conditions
        self.airspeed = 15.0        # Cruise airspeed (m/s)
        self.scaling_speed = 15.0   # Speed at which gains are tuned (m/s)
        
        # Base dynamics (will be randomized)
        self._set_base_dynamics()
        
        # State
        self.roll_angle = 0.0       # degrees
        self.roll_rate = 0.0        # deg/s
        
        # Control inputs (servo deflection in centidegrees)
        self.aileron_demand = 0.0   # centidegrees
        
    def _set_base_dynamics(self):
        """Set base aircraft dynamics (typical trainer aircraft at cruise)."""
        # Roll axis dynamics
        # Positive aileron (right stick) → right aileron down → aircraft rolls RIGHT → positive roll rate
        self.roll_damping = 3.0         # Natural roll damping (1/s)
        self.roll_control_power = 15.0  # Roll acceleration per aileron deflection (deg/s² per 100cd)
        self.roll_inertia = 1.0         # Relative inertia
        
        # Actuator dynamics
        self.aileron_rate_limit = 200   # deg/s (servo speed)
        self.elevator_rate_limit = 180  # deg/s
        self.aileron_pos = 0.0          # Current aileron position (centidegrees)
        self.elevator_pos = 0.0         # Current elevator position (centidegrees)
        self.aileron_max = 4500         # Max aileron deflection (centidegrees, ±45°)
        self.elevator_max = 4000        # Max elevator deflection (centidegrees, ±40°)
        
        # Store base values for randomization reference (roll only)
        self._base_params = {
            'roll_damping': self.roll_damping,
            'roll_control_power': self.roll_control_power,
        }
        
    def randomize(self, variation=0.2):
        """
        Randomize aircraft dynamics to simulate different airframes.

        Args:
            variation: Maximum variation from base (0.2 = ±20%)
        """
        for param in self._base_params:
            base = self._base_params[param]
            # Random variation between -variation and +variation
            factor = 1.0 + random.uniform(-variation, variation)
            setattr(self, param, base * factor)
            print(f"  {param}: {base:.2f} → {getattr(self, param):.2f} (×{factor:.2f})")
            
    def reset(self):
        """Reset aircraft state to level flight."""
        self.roll_angle = 0.0
        self.roll_rate = 0.0
        self.aileron_demand = 0.0
        self.aileron_pos = 0.0
        self.elevator_pos = 0.0
        
    def get_scaler(self):
        """Calculate airspeed scaler (matching ArduPilot)."""
        # scaler = scaling_speed / actual_airspeed
        return self.scaling_speed / max(self.airspeed, 1.0)
        
    def update(self, aileron_cd, elevator_cd):
        """
        Update aircraft dynamics.

        Args:
            aileron_cd: Aileron demand in centidegrees (-4500 to 4500)
            elevator_cd: Elevator demand in centidegrees (-4000 to 4000) - ignored
        """
        # Simulate actuator dynamics (first-order response)
        # ArduPilot servos typically have ~20ms response time
        actuator_tau = 0.02
        target_aileron = np.clip(aileron_cd, -self.aileron_max, self.aileron_max)

        self.aileron_pos += (target_aileron - self.aileron_pos) * self.dt / actuator_tau

        # Clip actuator positions
        self.aileron_pos = np.clip(self.aileron_pos, -self.aileron_max, self.aileron_max)

        # Calculate airspeed scaler for control authority
        scaler = self.get_scaler()

        # Control authority scales with 1/scaler² (matching ArduPilot inverse scaling)
        # When slower than scaling speed, controls are more effective
        control_scale = 1.0 / (scaler ** 2)

        # Roll dynamics: damping + control input
        # aileron_cd is in centidegrees, convert to "deflection units" (0-1)
        # Note: Positive aileron (right stick) produces positive roll rate (right wing down)
        aileron_deflection = self.aileron_pos / 100.0  # Convert to degrees equivalent
        roll_accel = (-self.roll_damping * self.roll_rate +
                      self.roll_control_power * aileron_deflection * control_scale) / self.roll_inertia

        # Integrate accelerations to rates
        self.roll_rate += roll_accel * self.dt

        # Integrate rates to angles
        self.roll_angle += self.roll_rate * self.dt

        # Store demands
        self.aileron_demand = aileron_cd


# ============================================================================
# Autotune Algorithm (matches ArduPilot AP_AutoTune)
# ============================================================================

class Autotuner:
    """
    Autotune algorithm for PIDFF parameters.
    Based on ArduPilot's AP_AutoTune implementation for fixed-wing.
    
    Strategy (matching ArduPilot):
    1. Apply step inputs and measure actuator/rate response
    2. Calculate FF from actuator/rate ratio (median filtered)
    3. Raise D gain until oscillation detected
    4. Set D_limit and raise P gain until oscillation
    5. Set final gains with I = FF/TRIM_TCONST
    
    Key ArduPilot constants:
    - AUTOTUNE_INCREASE_FF_STEP: 12%
    - AUTOTUNE_DECREASE_FF_STEP: 15%
    - TRIM_TCONST: 1.0
    - AUTOTUNE_I_RATIO: 0.75 (pitch), roll uses min(P, FF)
    """
    
    def __init__(self, pidff, aircraft, dt=0.02):
        self.pidff = pidff
        self.aircraft = aircraft
        self.dt = dt
        
        self.tuning = False
        self.tuning_axis = 'roll'
        self.tuning_phase = 0  # 0=not started, 1=FF, 2=D, 3=P, 4=done
        self.tuning_start_time = 0
        
        # State machine (matches ArduPilot ATState)
        self.state = 'IDLE'  # IDLE, DEMAND_POS, DEMAND_NEG
        self.state_enter_time = 0
        
        # Measurement buffers for step response
        self.step_outputs = []  # Servo outputs during step
        self.step_rates = []    # Achieved rates during step
        self.ff_estimates = []  # FF estimates (for median filter)
        
        # Gains during tuning
        self.initial_ff = 0.345
        self.initial_p = 0.08
        self.initial_d = 0.005
        
        # D and P limits when oscillation detected
        self.d_limit = 0.0
        self.p_limit = 0.0
        self.d_set_time = 0
        self.p_set_time = 0
        
        # Counters
        self.step_count = 0
        self.cycle_count = 0  # For phase 2/3 timing
        self.oscillation_detected = False
        
        # ArduPilot constants
        self.trim_tconst = 1.0
        self.i_ratio = 0.75
        self.min_imax = 0.4
        self.max_imax = 0.9
        
        # Step parameters - matching ArduPilot timing
        # Each step should take ~1.5-2 seconds (realistic aircraft response)
        self.step_amplitude = 30.0  # deg/s target rate
        self.step_duration = 1.5    # seconds per step direction (longer for realistic timing)
        self.min_event_time = 1.0   # Minimum time in each state (seconds)
        
        # Results
        self.tuned_params = {}
        
        # Current gains for display
        self.current_ff = 0.345
        self.current_p = 0.08
        self.current_i = 0.15
        self.current_d = 0.005
        
    def start_tuning(self, axis='roll'):
        """Start autotune process."""
        self.tuning = True
        self.tuning_axis = axis
        self.tuning_phase = 1  # Start with FF measurement
        self.state = 'IDLE'
        self.state_enter_time = 0
        self.step_outputs = []
        self.step_rates = []
        self.ff_estimates = []
        self.step_count = 0
        self.cycle_count = 0
        self.oscillation_detected = False
        self.oscillation_count = 0
        self.d_limit = 0.0
        self.p_limit = 0.0
        self.tuning_start_time = 0
        self.phase_start_time = 0  # Reset phase timing
        
        # Save initial gains
        self.initial_ff = self.pidff.ff_gain
        self.initial_p = self.pidff.p_gain
        self.initial_d = max(self.pidff.d_gain, 0.005)  # Ensure minimum D
        
        # Reset PIDFF for clean tuning
        self.pidff.reset()
        self.aircraft.reset()
        
        # Start with known gains
        self.pidff.ff_gain = self.initial_ff
        self.pidff.p_gain = self.initial_p
        self.pidff.d_gain = self.initial_d
        
        # Update current gains display
        self.current_ff = self.initial_ff
        self.current_p = self.initial_p
        self.current_i = self.pidff.i_gain
        self.current_d = self.initial_d
        
    def stop_tuning(self):
        """Stop autotune and apply results."""
        self.tuning = False
        if self.tuned_params:
            self.pidff.set_from_dict(self.tuned_params)
            
    def get_status(self):
        """Return current tuning status string with gains."""
        if not self.tuning:
            return "Not tuning"
            
        # Calculate time in current phase
        phase_elapsed = 0
        if hasattr(self, 'phase_start_time') and self.phase_start_time > 0:
            phase_elapsed = self.tuning_start_time - self.phase_start_time
            
        # ArduPilot timing:
        # Phase 1: 5 steps × 3s each = ~15s (IDLE→POS→NEG cycle = 3s)
        # Phase 2: 8s minimum + oscillation detection
        # Phase 3: 8s minimum + oscillation detection
        # Total: ~35-45 seconds
        phase_names = {
            0: "Initializing",
            1: f"Phase 1: FF (steps={self.step_count}/5, {phase_elapsed:.0f}s)",
            2: f"Phase 2: D (D={self.current_d:.4f}, {max(0, 8-phase_elapsed):.0f}s min)",
            3: f"Phase 3: P (P={self.current_p:.4f}, {max(0, 8-phase_elapsed):.0f}s min)",
            4: "Phase 4: Done"
        }
        phase_str = phase_names.get(self.tuning_phase, "Unknown")
        return f"{phase_str} FF={self.current_ff:.3f}"
        
    def update(self, target_rate, actual_rate, time):
        """
        Run autotune algorithm (matches ArduPilot AP_AutoTune::update).
        
        ArduPilot timing:
        - Each stick event takes 1.5-2 seconds minimum
        - State transitions are time-based (min 100ms, typically 500ms+)
        - FF measurement needs 5-7 events (~10-15 seconds)
        - D tuning: ~10 seconds minimum
        - P tuning: ~10 seconds minimum
        - Total autotune: ~30-40 seconds
        
        Returns:
            Control output (centidegrees) during tuning, or None if not tuning
        """
        if not self.tuning:
            return None
            
        if self.tuning_start_time == 0:
            self.tuning_start_time = time
            self.phase_start_time = time
        else:
            self.tuning_start_time = time  # Keep updated for phase elapsed calculation
            
        # State machine for step generation (matches ArduPilot ATState)
        # Timing is critical - each event should take 1.5-2 seconds
        if self.state == 'IDLE':
            if time - self.state_enter_time > self.min_event_time:  # Min 1s idle
                # Start positive step
                self.state = 'DEMAND_POS'
                self.state_enter_time = time
                self.step_outputs = []
                self.step_rates = []
                
        elif self.state == 'DEMAND_POS':
            # Must stay in this state for minimum time (aircraft needs time to respond)
            if time - self.state_enter_time > self.step_duration:  # 1.5s
                # Calculate FF from this step
                if self.step_outputs and self.step_rates:
                    max_output = max(self.step_outputs)
                    max_rate = max(self.step_rates)
                    if max_rate > 5.0:  # Need meaningful rate
                        ff_estimate = max_output / (max_rate * 100)  # Scale for centidegrees
                        if 0.05 < ff_estimate < 2.0:  # Sanity check
                            self.ff_estimates.append(ff_estimate)
                            self.step_count += 1
                
                # Go to negative step
                self.state = 'DEMAND_NEG'
                self.state_enter_time = time
                self.step_outputs = []
                self.step_rates = []
                
        elif self.state == 'DEMAND_NEG':
            # Must stay in this state for minimum time
            if time - self.state_enter_time > self.step_duration:  # 1.5s
                # Calculate FF from negative step
                if self.step_outputs and self.step_rates:
                    min_output = min(self.step_outputs)
                    min_rate = min(self.step_rates)
                    if min_rate < -5.0:
                        ff_estimate = min_output / (min_rate * 100)
                        if 0.05 < ff_estimate < 2.0:  # Sanity check
                            self.ff_estimates.append(ff_estimate)
                            self.step_count += 1
                
                # Go back to idle
                self.state = 'IDLE'
                self.state_enter_time = time
                
        # Collect output/rate data during steps
        if self.state != 'IDLE' and self.pidff.last_output != 0:
            self.step_outputs.append(abs(self.pidff.last_output))
            self.step_rates.append(abs(actual_rate))
            
        # Determine target rate based on state
        if self.state == 'DEMAND_POS':
            target_rate = self.step_amplitude
        elif self.state == 'DEMAND_NEG':
            target_rate = -self.step_amplitude
        else:
            target_rate = 0.0
            
        # Normal PIDFF control
        scaler = self.aircraft.get_scaler()
        control_output = self.pidff.update(target_rate, actual_rate, scaler=scaler)
        
        # Update current gains for display
        self.current_ff = self.pidff.ff_gain
        self.current_p = self.pidff.p_gain
        self.current_i = self.pidff.i_gain
        self.current_d = self.pidff.d_gain
        
        # Track time in each phase for minimum duration
        if not hasattr(self, 'phase_start_time'):
            self.phase_start_time = time
            
        # Phase-based tuning logic (with realistic timing like ArduPilot)
        if self.tuning_phase == 1:
            # Phase 1: Measure FF (need ~5 valid steps for median filter)
            # Each complete cycle (IDLE→POS→NEG→IDLE) takes ~4 seconds
            # 5 cycles = ~20 seconds for FF measurement
            if self.step_count >= 5 and len(self.ff_estimates) >= 5:
                # Apply median filter to FF estimates
                sorted_ff = sorted(self.ff_estimates)
                median_ff = sorted_ff[len(sorted_ff) // 2]
                
                # Limit FF change (matches ArduPilot)
                ff_change_limit = 0.12  # 12% increase max
                new_ff = min(median_ff, self.initial_ff * (1 + ff_change_limit))
                new_ff = max(new_ff, self.initial_ff * (1 - 0.15))  # 15% decrease max
                new_ff = max(new_ff, 0.1)  # Minimum FF
                
                self.pidff.ff_gain = new_ff
                self.current_ff = new_ff
                
                # Halve P to prepare for D tuning
                self.pidff.p_gain = self.initial_p * 0.5
                self.current_p = self.pidff.p_gain
                
                self.tuning_phase = 2
                self.step_count = 0
                self.cycle_count = 0
                self.phase_start_time = time
                self.oscillation_detected = False
                self.oscillation_count = 0  # Require multiple detections
                
        elif self.tuning_phase == 2:
            # Phase 2: Raise D until oscillation
            # ArduPilot minimum time: ~10 seconds for D tuning
            self.cycle_count += 1
            
            # Raise D by 10% every 1 second (slower, more realistic)
            if self.cycle_count % 50 == 0 and not self.oscillation_detected:  # ~1s at 50Hz
                self.pidff.d_gain *= 1.10
                self.current_d = self.pidff.d_gain
                
                # Check for oscillation (D term dominating)
                # Need sustained oscillation, not just one spike
                if abs(self.pidff.last_d) > abs(self.pidff.last_p) * 1.2:
                    self.oscillation_count += 1
                    
                # Require 3 consecutive detections (about 1s of oscillation)
                if self.oscillation_count >= 3 and not self.oscillation_detected:
                    self.oscillation_detected = True
                    self.d_limit = self.pidff.d_gain * 0.5  # Set D limit to 50% of current
                    self.pidff.d_gain = self.d_limit
                    self.current_d = self.d_limit
                    self.d_set_time = time
                    
            # Minimum 8 seconds in phase 2, or until oscillation + 2s settling
            min_phase2_time = 8.0
            if self.oscillation_detected:
                if time - self.d_set_time > 2.0:
                    self.tuning_phase = 3
                    self.step_count = 0
                    self.cycle_count = 0
                    self.phase_start_time = time
                    self.p_limit = 0
            elif time - self.phase_start_time > min_phase2_time:
                # If no oscillation after 8s, use current D and move on
                self.d_limit = self.pidff.d_gain
                self.tuning_phase = 3
                self.step_count = 0
                self.cycle_count = 0
                self.phase_start_time = time
                self.p_limit = 0
                
        elif self.tuning_phase == 3:
            # Phase 3: Raise P until oscillation
            # ArduPilot minimum time: ~10 seconds for P tuning
            self.cycle_count += 1
            
            # Raise P by 10% every 1 second (slower, more realistic)
            if self.cycle_count % 50 == 0 and self.p_limit == 0:
                self.pidff.p_gain *= 1.10
                self.current_p = self.pidff.p_gain
                
                # Check for oscillation (P term causing instability)
                if abs(self.pidff.last_p) > abs(self.pidff.last_ff) * 0.8:
                    self.oscillation_count += 1
                else:
                    self.oscillation_count = max(0, self.oscillation_count - 1)
                    
                # Require 3 consecutive detections
                if self.oscillation_count >= 3 and self.p_limit == 0:
                    self.p_limit = self.pidff.p_gain * 0.5
                    self.pidff.p_gain = self.p_limit
                    self.current_p = self.p_limit
                    self.p_set_time = time
                    
            # Minimum 8 seconds in phase 3, or until oscillation + 2s settling
            min_phase3_time = 8.0
            if self.p_limit > 0:
                if time - self.p_set_time > 2.0:
                    self.tuning_phase = 4
            elif time - self.phase_start_time > min_phase3_time:
                # If no oscillation after 8s, use current P
                self.p_limit = self.pidff.p_gain
                self.tuning_phase = 4
                
        elif self.tuning_phase == 4:
            # Phase 4: Done - calculate final gains
            ff_final = self.pidff.ff_gain
            p_final = self.pidff.p_gain
            
            # I gain based on FF and TRIM_TCONST (matches ArduPilot)
            if self.tuning_axis == 'roll':
                i_final = min(p_final, ff_final / self.trim_tconst)
            else:
                i_final = max(p_final * self.i_ratio, ff_final / self.trim_tconst)
                
            # Clamp IMAX
            imax = np.clip(i_final, self.min_imax, self.max_imax)
            
            self.tuned_params = {
                '_RATE_FF': ff_final,
                '_RATE_P': p_final,
                '_RATE_I': i_final,
                '_RATE_D': self.d_limit if self.d_limit > 0 else self.pidff.d_gain,
                '_RATE_IMAX': imax,
            }
            
            # Apply gains
            self.pidff.set_from_dict(self.tuned_params)
            self.current_ff = ff_final
            self.current_p = p_final
            self.current_i = i_final
            self.current_d = self.tuned_params['_RATE_D']
            
            self.tuning = False
            
        return control_output


# ============================================================================
# Scientific Tuner - Critical Damping Method
# ============================================================================

class ScientificTuner:
    """
    Scientific PID tuning for critically damped response.
    
    Uses system identification + pole placement:
    1. Apply step input and measure system response
    2. Identify system parameters (gain, time constant, damping)
    3. Calculate PID gains for critical damping (ζ = 1.0)
    
    For a 2nd order system: s² + 2ζωₙs + ωₙ²
    Critical damping: ζ = 1.0 (no overshoot, fastest non-oscillating)
    
    PID tuning rules for critical damping:
    - P = ωₙ² / K (where K is system gain)
    - D = 2ζωₙ / K = 2ωₙ / K (for ζ = 1)
    - I = small value for steady-state error correction
    - FF = 1/K (inverse of system gain)
    """
    
    def __init__(self, pidff, aircraft, dt=0.02):
        self.pidff = pidff
        self.aircraft = aircraft
        self.dt = dt
        
        self.tuning = False
        self.tuning_phase = 0
        self.tuning_start_time = 0
        
        # System identification data
        self.step_response = []  # (time, rate) pairs
        self.step_start_rate = 0.0
        self.step_input = 0.0
        
        # Identified parameters
        self.system_gain = 0.0      # K: steady-state rate / input
        self.time_constant = 0.0    # τ: time to reach 63% of steady-state
        self.delay = 0.0            # L: transport delay
        
        # Results
        self.tuned_params = {}
        
        # Current gains for display (matching Autotuner interface)
        self.current_ff = 0.345
        self.current_p = 0.08
        self.current_i = 0.15
        self.current_d = 0.005
        
    def start_tuning(self):
        """Start scientific tuning process."""
        self.tuning = True
        self.tuning_phase = 1  # System identification
        self.tuning_start_time = 0
        self.phase2_start_time = 0  # Will be set when phase 2 starts
        self.step_response = []
        self.step_start_rate = 0.0
        self.step_input = 2000  # centidegrees (~20° aileron)
        
        # Reset for clean measurement
        self.pidff.reset()
        self.aircraft.reset()
        
        # Use fixed gains for identification (conservative)
        self.pidff.ff_gain = 0.0
        self.pidff.p_gain = 0.05
        self.pidff.i_gain = 0.0
        self.pidff.d_gain = 0.0
        
    def stop_tuning(self):
        """Stop tuning and apply results."""
        self.tuning = False
        if self.tuned_params:
            self.pidff.set_from_dict(self.tuned_params)
            
    def get_status(self):
        """Return current tuning status."""
        if not self.tuning:
            return "Not tuning"
        phases = {
            1: "Phase 1: System ID (step response)",
            2: "Phase 2: Verify gains",
            3: "Phase 3: Complete"
        }
        return phases.get(self.tuning_phase, "Unknown")
        
    def update(self, target_rate, actual_rate, time):
        """
        Run scientific tuning algorithm.
        
        Phase transitions based on step response data:
        - Phase 1 → Phase 2: When step response reaches steady-state (rate change < 2 deg/s for 0.3s)
        - Phase 2 → Phase 3: After 0.5 seconds
        - Phase 3: Done, tuning complete
        
        Returns control output or None if not tuning.
        """
        if not self.tuning:
            return None
            
        if self.tuning_start_time == 0:
            self.tuning_start_time = time
            self.step_start_rate = actual_rate
            print(f"\n=== Scientific Tuner Started at t={time:.3f}s ===")
            
        # Debug: print phase info every 0.5s
        if int(time * 2) % 25 == 0:
            print(f"[t={time:.2f}s] Phase {self.tuning_phase}, tuning={self.tuning}")
            
        if self.tuning_phase == 1:
            # Phase 1: System identification via step response
            elapsed = time - self.tuning_start_time
            
            # Apply step input
            if elapsed < 0.05:
                target_rate = 0.0
            elif elapsed < 5.0:  # Up to 5 second step
                target_rate = 30.0  # deg/s target
            else:
                target_rate = 30.0  # Keep target until we detect steady-state
                
            # Record step response
            self.step_response.append((elapsed, actual_rate))
            
            # Debug: print every 0.5s
            if len(self.step_response) % 25 == 0:
                print(f"Phase 1: t={elapsed:.2f}s, rate={actual_rate:.1f} deg/s")
            
            # Check if we've reached steady-state
            # (rate change < 2 deg/s over last 0.3 seconds = 15 samples at 50Hz)
            if len(self.step_response) >= 20:
                recent_rates = [r[1] for r in self.step_response[-15:]]
                rate_change = max(recent_rates) - min(recent_rates)
                
                if rate_change < 2.0:  # Steady-state reached
                    print(f"\nSteady-state detected at t={elapsed:.2f}s (rate variation: {rate_change:.2f} deg/s)")
                    self._identify_system()
                    self._calculate_gains()
                    self.phase2_start_time = time  # Mark when phase 2 starts
                    print(f"Phase 2 started: time={time:.3f}, phase2_start_time={self.phase2_start_time:.3f}")
                    
            # Fallback: force transition after 5 seconds even if no steady-state
            if elapsed > 5.0 and self.tuning_phase == 1:
                print(f"\nFallback: forcing phase transition at t={elapsed:.2f}s")
                self._identify_system()
                self._calculate_gains()
                self.phase2_start_time = time
                    
        elif self.tuning_phase == 2:
            # Phase 2: Brief verification (0.5 seconds)
            # Just let the new gains settle
            phase2_elapsed = time - self.phase2_start_time
            target_rate = 0.0
            
            # Debug: print phase 2 timing
            print(f"  Phase 2: time={time:.3f}, phase2_start={self.phase2_start_time:.3f}, elapsed={phase2_elapsed:.3f}s")
            
            if phase2_elapsed > 0.5:
                print(f"Phase 2 complete ({phase2_elapsed:.2f}s), moving to Phase 3")
                self.tuning_phase = 3
            
        elif self.tuning_phase == 3:
            # Phase 3: Done
            print("Phase 3: Tuning complete!")
            self.tuning = False
            target_rate = 0.0
            
        # Normal control during tuning
        scaler = self.aircraft.get_scaler()
        return self.pidff.update(target_rate, actual_rate, scaler=scaler)
        
    def _identify_system(self):
        """Identify system parameters from step response."""
        if len(self.step_response) < 10:
            # Fallback defaults
            self.system_gain = 2.0
            self.time_constant = 0.3
            self.delay = 0.05
            return
            
        # Find steady-state value (average of last 20% of response)
        n_samples = len(self.step_response)
        steady_samples = self.step_response[int(n_samples * 0.8):]
        steady_rate = np.mean([r[1] for r in steady_samples])
        
        # System gain K = steady_state_output / input
        # Input is in centidegrees, output in deg/s
        self.system_gain = steady_rate / self.step_input  # deg/s per cd
        
        # Find time constant (time to reach 63.2% of steady-state)
        target_63 = self.step_start_rate + 0.632 * (steady_rate - self.step_start_rate)
        self.time_constant = 0.3  # Default
        
        for t, rate in self.step_response:
            if rate >= target_63:
                self.time_constant = t
                break
                
        # Find delay (time to reach 10% of response)
        target_10 = self.step_start_rate + 0.1 * (steady_rate - self.step_start_rate)
        self.delay = 0.02  # Default
        
        for t, rate in self.step_response:
            if rate >= target_10:
                self.delay = t
                break
                
        print(f"\n=== System Identification ===")
        print(f"Steady-state rate: {steady_rate:.1f} deg/s")
        print(f"System gain K: {self.system_gain*1000:.2f} deg/s per 100cd")
        print(f"Time constant τ: {self.time_constant:.3f} s")
        print(f"Delay L: {self.delay:.3f} s")
        
    def _calculate_gains(self):
        """Calculate PID gains for critically damped response."""
        # For critically damped 2nd order system (ζ = 1.0):
        # Natural frequency ωₙ determines response speed
        # Target: fast response without overshoot
        
        # Choose natural frequency based on time constant
        # ωₙ = 1/τ gives response time ~4τ (settling to 2%)
        omega_n = 1.0 / max(self.time_constant, 0.1)
        
        # Limit to reasonable values (conservative for stability)
        omega_n = np.clip(omega_n, 1.0, 5.0)  # 1-5 rad/s
        
        # System gain K (deg/s per centidegree)
        K = self.system_gain  # Already in deg/s per cd
        
        # PID gains for critical damping with stability margin:
        # FF = 1/K (for perfect tracking)
        # P = ωₙ² / K (but limited for stability)
        # D = 2ζωₙ / K reduced for actuator lag compensation
        
        # FF gain: inverse of system gain
        ff_gain = 1.0 / max(K * 100, 0.01)
        
        # P gain: reduced from theoretical for stability
        p_gain = (omega_n ** 2) / max(K * 100, 0.01) * 0.3  # 30% of theoretical
        
        # D gain: heavily reduced to account for actuator lag and filtering
        # High D causes oscillation with delayed systems
        d_gain = (2.0 * omega_n) / max(K * 100, 0.01) * 0.08  # 8% of theoretical
        
        # I gain: small value for steady-state correction
        i_gain = 0.02 * p_gain
        
        # IMAX for integral limiting
        imax = 0.5
        
        self.tuned_params = {
            '_RATE_FF': ff_gain,
            '_RATE_P': p_gain,
            '_RATE_I': i_gain,
            '_RATE_D': d_gain,
            '_RATE_IMAX': imax,
        }
        
        print(f"\n=== Calculated Gains (Conservative Critical Damping) ===")
        print(f"Natural frequency ωₙ: {omega_n:.1f} rad/s")
        print(f"System gain K: {K*100:.2f} deg/s per 100cd")
        print(f"FF: {ff_gain:.4f} (1/K)")
        print(f"P:  {p_gain:.4f} (30% of theoretical)")
        print(f"I:  {i_gain:.4f}")
        print(f"D:  {d_gain:.4f} (8% for actuator lag)")
        print(f"Expected settling time: {4/omega_n:.2f} s")
        
        # Apply gains directly (no limits)
        self.pidff.ff_gain = ff_gain
        self.pidff.p_gain = p_gain
        self.pidff.i_gain = i_gain
        self.pidff.d_gain = d_gain
        self.pidff.imax = imax
        
        # Update current gains for display
        self.current_ff = ff_gain
        self.current_p = p_gain
        self.current_i = i_gain
        self.current_d = d_gain
        
        print(f"\nGains applied to controller!")
        
        # Update spinboxes in GUI (via main window callback)
        GLib.idle_add(self._update_spinboxes)
        
        # Force phase transition after gains calculated
        self.tuning_phase = 2
        
    def _update_spinboxes(self):
        """Update parameter spinboxes (called via GLib.idle_add)."""
        # This will be overridden by the window instance
        pass


# ============================================================================
# GTK Application
# ============================================================================

class PIDFFSimulatorWindow(Adw.ApplicationWindow):
    """Main application window."""
    
    def __init__(self, app):
        super().__init__(application=app)
        
        self.set_title("ArduPilot PIDFF Simulator")
        self.set_default_size(1400, 900)
        
        # Simulation parameters (matching ArduPlane main loop ~50Hz)
        self.dt = 0.02  # 50Hz
        self.simulation_time = 0.0
        self.running = False
        self.step_mode = None  # 'positive', 'negative', or None
        self.step_start_time = 0
        
        # Data buffers for plotting (last 5 seconds)
        self.buffer_size = 500
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.target_buffer = deque(maxlen=self.buffer_size)
        self.actual_buffer = deque(maxlen=self.buffer_size)
        self.p_buffer = deque(maxlen=self.buffer_size)
        self.i_buffer = deque(maxlen=self.buffer_size)
        self.d_buffer = deque(maxlen=self.buffer_size)
        self.ff_buffer = deque(maxlen=self.buffer_size)
        self.error_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize controllers and aircraft
        self.roll_pidff = ArduPilotPIDFF(dt=self.dt, axis='roll')
        self.pitch_pidff = ArduPilotPIDFF(dt=self.dt, axis='pitch')
        self.aircraft = AircraftDynamics(dt=self.dt)
        self.roll_autotuner = Autotuner(self.roll_pidff, self.aircraft, dt=self.dt)
        self.pitch_autotuner = Autotuner(self.pitch_pidff, self.aircraft, dt=self.dt)
        self.scientific_tuner = ScientificTuner(self.roll_pidff, self.aircraft, dt=self.dt)
        
        # Current axis being tuned
        self.current_axis = 'roll'
        self._was_scientific_tuning = False
        
        # Build UI - use set_content() for AdwApplicationWindow
        main_box = self._build_ui()
        self.set_content(main_box)
        
        # Add keyboard shortcut for spacebar (triggers +45° step)
        key_controller = Gtk.EventControllerKey()
        key_controller.connect('key-pressed', self._on_key_pressed)
        self.add_controller(key_controller)
        
        # Start simulation loop
        GLib.timeout_add(int(self.dt * 1000), self._simulation_step)
        
    def _build_ui(self):
        """Build the user interface."""
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        
        # Header with close button support
        header = Adw.HeaderBar()
        header.set_show_title(True)
        header.set_title_widget(Gtk.Label(label="<b>ArduPilot PIDFF Simulator</b> - Fixed-Wing Roll at Cruise"))
        main_box.append(header)
        
        # Info label below header
        info_label = Gtk.Label()
        info_label.set_markup("<b>ArduPilot PIDFF Simulator</b> - Fixed-Wing Roll at Cruise (APM_Control)")
        info_label.set_margin_bottom(10)
        main_box.append(info_label)
        
        # Control buttons row
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        control_box.set_halign(Gtk.Align.CENTER)
        main_box.append(control_box)
        
        # Start/Stop button
        self.start_button = Gtk.Button(label="▶ Start")
        self.start_button.connect('clicked', self._on_start_stop)
        control_box.append(self.start_button)
        
        # Step input buttons
        step_label = Gtk.Label(label="Step Input:")
        step_label.set_margin_start(20)
        control_box.append(step_label)
        
        self.step_pos_button = Gtk.Button(label="+45° Roll Step")
        self.step_pos_button.connect('clicked', self._on_step_positive)
        control_box.append(self.step_pos_button)
        
        self.step_neg_button = Gtk.Button(label="-45° Roll Step")
        self.step_neg_button.connect('clicked', self._on_step_negative)
        control_box.append(self.step_neg_button)
        
        # Autotune button
        self.autotune_button = Gtk.Button(label="🔧 Autotune (AP)")
        self.autotune_button.connect('clicked', self._on_autotune)
        control_box.append(self.autotune_button)
        
        # Scientific tune button (Ziegler-Nichols / Critical Damping)
        self.scientune_button = Gtk.Button(label="📐 Critical Damping Tune")
        self.scientune_button.connect('clicked', self._on_scientune)
        control_box.append(self.scientune_button)
        
        # Randomize button
        self.randomize_button = Gtk.Button(label="🎲 Randomize Aircraft")
        self.randomize_button.connect('clicked', self._on_randomize)
        control_box.append(self.randomize_button)
        
        # Reset button
        self.reset_button = Gtk.Button(label="🔄 Reset")
        self.reset_button.connect('clicked', self._on_reset)
        control_box.append(self.reset_button)
        
        # Graphs area using Gtk.DrawingArea
        graphs_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        graphs_box.set_margin_top(10)
        graphs_box.set_margin_bottom(10)
        main_box.append(graphs_box)
        
        # Response graph
        response_frame = Gtk.Frame(label="Step Response (deg/s)")
        self.response_drawing = Gtk.DrawingArea()
        self.response_drawing.set_size_request(650, 280)
        self.response_drawing.set_draw_func(self._draw_response_graph)
        response_frame.set_child(self.response_drawing)
        graphs_box.append(response_frame)
        
        # PIDFF components graph
        pidff_frame = Gtk.Frame(label="PIDFF Components (servo centidegrees)")
        self.pidff_drawing = Gtk.DrawingArea()
        self.pidff_drawing.set_size_request(650, 280)
        self.pidff_drawing.set_draw_func(self._draw_pidff_graph)
        pidff_frame.set_child(self.pidff_drawing)
        graphs_box.append(pidff_frame)
        
        # Parameter controls
        params_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        params_box.set_margin_top(10)
        main_box.append(params_box)
        
        # Roll parameters (only roll axis simulated)
        roll_frame = self._create_param_frame("Roll Axis (SRV2SRV_*)", self.roll_pidff, 'roll')
        params_box.append(roll_frame)
        
        # Info panel
        info_frame = self._create_info_frame()
        params_box.append(info_frame)
        
        # Autotune status panel
        autotune_frame = self._create_autotune_frame()
        params_box.append(autotune_frame)
        
        # Status bar
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        status_box.set_margin_top(10)
        main_box.append(status_box)
        
        self.status_label = Gtk.Label(label="Status: Stopped")
        status_box.append(self.status_label)
        
        self.error_label = Gtk.Label(label="Error: 0.0 deg/s")
        status_box.append(self.error_label)
        
        self.aircraft_label = Gtk.Label(label="Aircraft: Standard")
        status_box.append(self.aircraft_label)
        
        self.scaler_label = Gtk.Label(label="Scaler: 1.0")
        status_box.append(self.scaler_label)
        
        return main_box
        
    def _create_param_frame(self, title, pidff, axis):
        """Create parameter adjustment frame with ArduPilot parameters."""
        frame = Gtk.Frame(label=title)
        frame.set_size_request(400, 220)  # Wider frame
        
        grid = Gtk.Grid()
        grid.set_margin_start(10)
        grid.set_margin_end(10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_row_spacing(5)
        grid.set_column_spacing(10)
        frame.set_child(grid)
        
        # ArduPilot APM_Control parameters (wide ranges for tuning)
        params = [
            ('_RATE_P', 0.001, 2.0, 0.001),
            ('_RATE_I', 0.001, 2.0, 0.001),
            ('_RATE_D', 0.0, 0.5, 0.001),
            ('_RATE_FF', 0.01, 5.0, 0.01),
            ('_RATE_IMAX', 0.1, 2.0, 0.05),
            ('_RATE_FLTT', 1, 50, 1),
        ]
        
        self.param_spinboxes = {}
        
        for row, (name, min_val, max_val, step) in enumerate(params):
            label = Gtk.Label(label=f"{name}:")
            label.set_halign(Gtk.Align.START)
            grid.attach(label, 0, row, 1, 1)
            
            # Get current value
            attr_map = {
                '_RATE_P': 'p_gain', '_RATE_I': 'i_gain', '_RATE_D': 'd_gain',
                '_RATE_FF': 'ff_gain', '_RATE_IMAX': 'imax', '_RATE_FLTT': 'target_filter_hz'
            }
            current_value = getattr(pidff, attr_map[name])
            
            adjustment = Gtk.Adjustment(
                value=current_value,
                lower=min_val,
                upper=max_val,
                step_increment=step,
                page_increment=step * 5
            )
            
            spin = Gtk.SpinButton(adjustment=adjustment)
            spin.set_digits(3 if step < 0.1 else 0)
            spin.set_name(f"{axis}_{name}")
            spin.set_size_request(150, -1)  # Much wider spinboxes
            spin.connect('value-changed', self._on_param_changed, axis, name)
            grid.attach(spin, 1, row, 1, 1)
            
            self.param_spinboxes[f"{axis}_{name}"] = spin
            
        return frame

    def _create_info_frame(self):
        """Create info panel with tuning tips."""
        frame = Gtk.Frame(label="Tuning Guide")
        frame.set_size_request(280, 220)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        frame.set_child(box)
        
        tips = [
            "<b>Tuning Order:</b>",
            "1. Set _RATE_FF first (0.3-0.5)",
            "2. Tune _RATE_P for response",
            "3. Add _RATE_I to eliminate error",
            "4. Add _RATE_D if oscillating",
            "",
            "<b>Typical Values:</b>",
            "Roll: P=0.08, FF=0.35",
            "",
            "<b>Scaler:</b> affects gain",
            "slower = more authority",
        ]
        
        for tip in tips:
            label = Gtk.Label()
            label.set_markup(tip)
            label.set_halign(Gtk.Align.START)
            box.append(label)
            
        return frame
        
    def _create_autotune_frame(self):
        """Create autotune status panel."""
        frame = Gtk.Frame(label="Autotune Status")
        frame.set_size_request(280, 220)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        frame.set_child(box)
        
        # Status label
        self.autotune_status_label = Gtk.Label(label="Not tuning")
        self.autotune_status_label.set_halign(Gtk.Align.START)
        self.autotune_status_label.set_markup("<b>Status:</b> Not tuning")
        box.append(self.autotune_status_label)
        
        # Separator
        box.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        
        # Current gains grid
        gains_grid = Gtk.Grid()
        gains_grid.set_row_spacing(4)
        gains_grid.set_column_spacing(10)
        box.append(gains_grid)
        
        # FF gain
        label = Gtk.Label(label="FF:")
        label.set_halign(Gtk.Align.START)
        gains_grid.attach(label, 0, 0, 1, 1)
        self.autotune_ff_label = Gtk.Label(label="0.345")
        self.autotune_ff_label.set_halign(Gtk.Align.END)
        gains_grid.attach(self.autotune_ff_label, 1, 0, 1, 1)
        
        # P gain
        label = Gtk.Label(label="P:")
        label.set_halign(Gtk.Align.START)
        gains_grid.attach(label, 0, 1, 1, 1)
        self.autotune_p_label = Gtk.Label(label="0.080")
        self.autotune_p_label.set_halign(Gtk.Align.END)
        gains_grid.attach(self.autotune_p_label, 1, 1, 1, 1)
        
        # I gain
        label = Gtk.Label(label="I:")
        label.set_halign(Gtk.Align.START)
        gains_grid.attach(label, 0, 2, 1, 1)
        self.autotune_i_label = Gtk.Label(label="0.150")
        self.autotune_i_label.set_halign(Gtk.Align.END)
        gains_grid.attach(self.autotune_i_label, 1, 2, 1, 1)
        
        # D gain
        label = Gtk.Label(label="D:")
        label.set_halign(Gtk.Align.START)
        gains_grid.attach(label, 0, 3, 1, 1)
        self.autotune_d_label = Gtk.Label(label="0.005")
        self.autotune_d_label.set_halign(Gtk.Align.END)
        gains_grid.attach(self.autotune_d_label, 1, 3, 1, 1)
        
        # Steps counter
        box.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self.autotune_steps_label = Gtk.Label(label="Steps: 0/5")
        self.autotune_steps_label.set_halign(Gtk.Align.START)
        box.append(self.autotune_steps_label)
        
        return frame
        
    def _update_autotune_display(self):
        """Update autotune status display."""
        if self.scientific_tuner.tuning:
            autotuner = self.scientific_tuner
        elif self.roll_autotuner.tuning:
            autotuner = self.roll_autotuner
        elif self.pitch_autotuner.tuning:
            autotuner = self.pitch_autotuner
        else:
            self.autotune_status_label.set_markup("<b>Status:</b> Not tuning")
            self.autotune_steps_label.set_label("Steps: 0/5")
            return
            
        # Update status
        status = autotuner.get_status()
        self.autotune_status_label.set_markup(f"<b>Status:</b> {status}")
        
        # Update gains
        self.autotune_ff_label.set_label(f"{autotuner.current_ff:.4f}")
        self.autotune_p_label.set_label(f"{autotuner.current_p:.4f}")
        self.autotune_i_label.set_label(f"{autotuner.current_i:.4f}")
        self.autotune_d_label.set_label(f"{autotuner.current_d:.4f}")
        
        # Update steps
        if hasattr(autotuner, 'step_count'):
            self.autotune_steps_label.set_label(f"Steps: {autotuner.step_count}/5 (Phase {autotuner.tuning_phase})")
        else:
            self.autotune_steps_label.set_label(f"Phase {autotuner.tuning_phase}")

    def _on_param_changed(self, spin, axis, param_name):
        """Handle parameter change."""
        value = spin.get_value()
        pidff = self.roll_pidff if axis == 'roll' else self.pitch_pidff
        
        attr_map = {
            '_RATE_P': 'p_gain', '_RATE_I': 'i_gain', '_RATE_D': 'd_gain',
            '_RATE_FF': 'ff_gain', '_RATE_IMAX': 'imax', '_RATE_FLTT': 'target_filter_hz',
            '_RATE_FLTE': 'error_filter_hz', '_RATE_FLTD': 'deriv_filter_hz',
        }
        
        if param_name in attr_map:
            setattr(pidff, attr_map[param_name], value)
            
    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle keyboard shortcuts."""
        # Space bar triggers +45° step
        if keyval == Gdk.KEY_space:
            self._start_step('positive')
            return True  # Consume the event
        return False  # Pass through other keys
        
    def _on_start_stop(self, button):
        """Toggle simulation start/stop."""
        self.running = not self.running
        if self.running:
            button.set_label("⏸ Stop")
            self.status_label.set_label("Status: Running")
        else:
            button.set_label("▶ Start")
            self.status_label.set_label("Status: Stopped")
            self.step_mode = None
            
    def _on_step_positive(self, button):
        """Start positive step sequence."""
        self._start_step('positive')
        
    def _on_step_negative(self, button):
        """Start negative step sequence."""
        self._start_step('negative')
        
    def _start_step(self, direction):
        """Start step input sequence."""
        # Reset controller and aircraft state for clean step response
        self.roll_pidff.reset()
        self.pitch_pidff.reset()
        self.aircraft.reset()
        
        # Clear buffers for fresh graph
        self.time_buffer.clear()
        self.target_buffer.clear()
        self.actual_buffer.clear()
        self.p_buffer.clear()
        self.i_buffer.clear()
        self.d_buffer.clear()
        self.ff_buffer.clear()
        self.error_buffer.clear()
        
        self.step_mode = direction
        self.step_start_time = self.simulation_time
        self.step_phase = 0  # 0: step up, 1: hold, 2: return
        
        if not self.running:
            self.running = True
            self.start_button.set_label("⏸ Stop")
        self.status_label.set_label(f"Status: Step Test ({direction})")
            
    def _on_autotune(self, button):
        """Start autotune for current axis."""
        if self.current_axis == 'roll':
            self.roll_autotuner.start_tuning('roll')
            self.status_label.set_label("Status: Autotuning Roll (AP method)...")
        else:
            self.pitch_autotuner.start_tuning('pitch')
            self.status_label.set_label("Status: Autotuning Pitch (AP method)...")
        
        self.running = True
        self.start_button.set_label("⏸ Stop")
        
    def _on_scientune(self, button):
        """Start scientific critical damping tune."""
        self.scientific_tuner.start_tuning()
        self.scientific_tuner._update_spinboxes = self._update_spinboxes_from_tuner
        self._was_scientific_tuning = True
        self.status_label.set_label("Status: Scientific Tune (Critical Damping)...")
        self.running = True
        self.start_button.set_label("⏸ Stop")
        
    def _update_spinboxes_from_tuner(self):
        """Update spinboxes after scientific tuning."""
        for name, attr in [('_RATE_FF', 'ff_gain'), ('_RATE_P', 'p_gain'),
                           ('_RATE_I', 'i_gain'), ('_RATE_D', 'd_gain'),
                           ('_RATE_IMAX', 'imax'), ('_RATE_FLTT', 'target_filter_hz')]:
            key = f"roll_{name}"
            if key in self.param_spinboxes:
                self.param_spinboxes[key].set_value(getattr(self.roll_pidff, attr))
        
    def _on_randomize(self, button):
        """Randomize aircraft dynamics with wide variation."""
        print("Randomizing aircraft dynamics (±70% variation)...")
        self.aircraft.randomize(variation=0.7)  # ±70% variation
        self.aircraft_label.set_label("Aircraft: Randomized")
        self.status_label.set_label("Status: Aircraft randomized (±70%) - retune PIDs!")
        print("Aircraft randomized - retune PIDs for new dynamics!")
        
    def _on_reset(self, button):
        """Reset simulation."""
        self.running = False
        self.start_button.set_label("▶ Start")
        self.step_mode = None
        
        self.roll_pidff.reset()
        self.pitch_pidff.reset()
        self.aircraft.reset()
        self.roll_autotuner.stop_tuning()
        self.pitch_autotuner.stop_tuning()
        self.scientific_tuner.stop_tuning()
        
        self.simulation_time = 0.0
        
        # Clear buffers
        self.time_buffer.clear()
        self.target_buffer.clear()
        self.actual_buffer.clear()
        self.p_buffer.clear()
        self.i_buffer.clear()
        self.d_buffer.clear()
        self.ff_buffer.clear()
        self.error_buffer.clear()
        
        self.status_label.set_label("Status: Reset")
        self.aircraft_label.set_label("Aircraft: Standard")
        
        # Reset parameter displays (roll only)
        for name, attr in [('_RATE_P', 'p_gain'), ('_RATE_I', 'i_gain'),
                           ('_RATE_D', 'd_gain'), ('_RATE_FF', 'ff_gain'),
                           ('_RATE_IMAX', 'imax'), ('_RATE_FLTT', 'target_filter_hz')]:
            key = f"roll_{name}"
            if key in self.param_spinboxes:
                self.param_spinboxes[key].set_value(getattr(self.roll_pidff, attr))
        
        # Force redraw
        self.response_drawing.queue_draw()
        self.pidff_drawing.queue_draw()
        
    def _simulation_step(self):
        """Main simulation loop."""
        try:
            if self.running:
                self.simulation_time += self.dt
                
                # Determine target rate based on step mode or autotune
                target_rate = 0.0
                
                if self.step_mode:
                    elapsed = self.simulation_time - self.step_start_time
                    if elapsed < 0.1:  # Small delay before step
                        target_rate = 0.0
                    elif elapsed < 2.1:  # 2 second hold
                        target_rate = 45.0 if self.step_mode == 'positive' else -45.0
                    else:  # Return to zero
                        target_rate = 0.0
                        self.step_mode = None
                        
                # Get current PIDFF (using roll axis)
                pidff = self.roll_pidff
                autotuner = self.roll_autotuner
                
                # Get airspeed scaler
                scaler = self.aircraft.get_scaler()
                
                # Check if scientific tuning (takes priority)
                if self.scientific_tuner.tuning:
                    control_output = self.scientific_tuner.update(
                        target_rate, self.aircraft.roll_rate, self.simulation_time)
                    # Update status every 0.2s
                    if self.simulation_time % 0.2 < self.dt:
                        self.status_label.set_label(f"Status: {self.scientific_tuner.get_status()}")
                else:
                    # Just finished scientific tuning - update status to show complete
                    if hasattr(self, '_was_scientific_tuning') and self._was_scientific_tuning:
                        self.status_label.set_label("Status: Scientific Tune Complete!")
                        self._was_scientific_tuning = False
                    # Normal PIDFF control (output in centidegrees)
                    control_output = pidff.update(target_rate, self.aircraft.roll_rate, scaler=scaler)
                
                # Update aircraft (control output in centidegrees)
                self.aircraft.update(control_output, 0.0)  # Only roll for now
                
                # Get values for storage (with NaN protection)
                roll_rate = self.aircraft.roll_rate
                if not np.isfinite(roll_rate):
                    roll_rate = 0.0
                    
                error = target_rate - roll_rate
                if not np.isfinite(error):
                    error = 0.0
                
                # Store data for plotting (with NaN protection)
                self.time_buffer.append(self.simulation_time)
                self.target_buffer.append(target_rate if np.isfinite(target_rate) else 0.0)
                self.actual_buffer.append(roll_rate)
                self.p_buffer.append(pidff.last_p if np.isfinite(pidff.last_p) else 0.0)
                self.i_buffer.append(pidff.last_i if np.isfinite(pidff.last_i) else 0.0)
                self.d_buffer.append(pidff.last_d if np.isfinite(pidff.last_d) else 0.0)
                self.ff_buffer.append(pidff.last_ff if np.isfinite(pidff.last_ff) else 0.0)
                self.error_buffer.append(error)
                
                # Update status
                self.error_label.set_label(f"Error: {error:.1f} deg/s")
                self.scaler_label.set_label(f"Scaler: {scaler:.2f}")
                
                # Update autotune display
                self._update_autotune_display()
                
                # Trigger graph redraw
                self.response_drawing.queue_draw()
                self.pidff_drawing.queue_draw()
        except Exception as e:
            # Log error but don't crash - just stop the simulation
            print(f"Simulation error: {e}")
            self.status_label.set_label(f"Status: Error - {str(e)[:50]}")
            self.running = False
            self.start_button.set_label("▶ Start")
            
        return True  # Continue timeout
        
    def _draw_response_graph(self, drawing, cr, width, height):
        """Draw step response graph."""
        # Background
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        
        # Check for valid dimensions
        if width < 100 or height < 100:
            return
            
        if len(self.time_buffer) < 2:
            return
        
        # Setup coordinate system
        margin_left = 50
        margin_right = 10
        margin_top = 20
        margin_bottom = 40
        
        plot_width = max(width - margin_left - margin_right, 10)
        plot_height = max(height - margin_top - margin_bottom, 10)
        
        # Calculate scales
        times = list(self.time_buffer)
        targets = list(self.target_buffer)
        actuals = list(self.actual_buffer)
        
        if not times or len(times) < 2:
            return
            
        t_min = times[0]
        t_max = times[-1]
        t_range = max(t_max - t_min, 0.1)
        
        # Filter out invalid values
        valid_targets = [v for v in targets if np.isfinite(v)]
        valid_actuals = [v for v in actuals if np.isfinite(v)]
        
        if not valid_targets or not valid_actuals:
            return
            
        all_values = valid_targets + valid_actuals
        v_min = min(all_values) if all_values else -50
        v_max = max(all_values) if all_values else 50
        
        # Ensure reasonable range
        if abs(v_max - v_min) < 1:
            v_max = v_min + 50
            
        v_range = max(v_max - v_min, 10)
        v_min -= v_range * 0.1
        v_max += v_range * 0.1
        v_range = v_max - v_min
        
        # Draw grid
        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.set_line_width(1)
        
        # Horizontal grid lines
        for i in range(5):
            y = margin_top + (plot_height * i / 4)
            cr.move_to(margin_left, y)
            cr.line_to(width - margin_right, y)
            cr.stroke()
            
            # Value labels
            value = v_max - (v_range * i / 4)
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.set_font_size(10)
            cr.move_to(margin_left - 45, y + 3)
            cr.show_text(f"{value:.0f}")
            
        # Vertical grid lines (time)
        for i in range(6):
            x = margin_left + (plot_width * i / 5)
            cr.move_to(x, margin_top)
            cr.line_to(x, height - margin_bottom)
            cr.stroke()
            
            # Time labels
            time_val = t_min + (t_range * i / 5)
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.move_to(x - 15, height - margin_bottom + 20)
            cr.show_text(f"{time_val:.1f}s")
            
        # Axis labels
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.set_font_size(12)
        cr.move_to(width / 2 - 30, height - 10)
        cr.show_text("Time (s)")
        
        # Helper function to calculate Y position safely
        def calc_y(val):
            if not np.isfinite(val) or v_range == 0:
                return margin_top + plot_height / 2
            return margin_top + (1 - (val - v_min) / v_range) * plot_height
            
        def calc_x(t):
            if t_range == 0:
                return margin_left
            return margin_left + (t - t_min) / t_range * plot_width
        
        # Draw target (dashed line)
        cr.set_source_rgb(0.2, 0.4, 0.8)
        cr.set_line_width(2)
        cr.set_dash([5, 5])
        
        cr.new_path()
        first_point = True
        for i, (t, val) in enumerate(zip(times, targets)):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        cr.set_dash([])
        
        # Draw actual response (solid line)
        cr.set_source_rgb(0.8, 0.2, 0.2)
        cr.set_line_width(2)
        
        cr.new_path()
        first_point = True
        for i, (t, val) in enumerate(zip(times, actuals)):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        
        # Legend
        cr.set_font_size(11)
        cr.set_source_rgb(0.2, 0.4, 0.8)
        cr.move_to(width - 120, 25)
        cr.show_text("─ Target")
        cr.set_source_rgb(0.8, 0.2, 0.2)
        cr.move_to(width - 120, 40)
        cr.show_text("─ Actual")
        
    def _draw_pidff_graph(self, drawing, cr, width, height):
        """Draw PIDFF components graph."""
        # Background
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        
        # Check for valid dimensions
        if width < 100 or height < 100:
            return
            
        if len(self.time_buffer) < 2:
            return
        
        # Setup coordinate system
        margin_left = 50
        margin_right = 10
        margin_top = 20
        margin_bottom = 40
        
        plot_width = max(width - margin_left - margin_right, 10)
        plot_height = max(height - margin_top - margin_bottom, 10)
        
        # Get data
        times = list(self.time_buffer)
        p_vals = list(self.p_buffer)
        i_vals = list(self.i_buffer)
        d_vals = list(self.d_buffer)
        ff_vals = list(self.ff_buffer)
        
        if not times or len(times) < 2:
            return
        
        t_min = times[0]
        t_max = times[-1]
        t_range = max(t_max - t_min, 0.1)
        
        # Filter out invalid values
        all_values = [v for v in (p_vals + i_vals + d_vals + ff_vals) if np.isfinite(v)]
        
        if not all_values:
            return
            
        v_min = min(all_values) if all_values else -1
        v_max = max(all_values) if all_values else 1
        
        # Ensure reasonable range
        if abs(v_max - v_min) < 0.1:
            v_max = v_min + 1
            
        v_range = max(v_max - v_min, 0.1)
        v_min -= v_range * 0.1
        v_max += v_range * 0.1
        v_range = v_max - v_min
        
        # Draw grid
        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.set_line_width(1)
        
        for i in range(5):
            y = margin_top + (plot_height * i / 4)
            cr.move_to(margin_left, y)
            cr.line_to(width - margin_right, y)
            cr.stroke()
            
        for i in range(6):
            x = margin_left + (plot_width * i / 5)
            cr.move_to(x, margin_top)
            cr.line_to(x, height - margin_bottom)
            cr.stroke()
            
        # Helper function to calculate Y position safely
        def calc_y(val):
            if not np.isfinite(val) or v_range == 0:
                return margin_top + plot_height / 2
            return margin_top + (1 - (val - v_min) / v_range) * plot_height
            
        def calc_x(t):
            if t_range == 0:
                return margin_left
            return margin_left + (t - t_min) / t_range * plot_width
        
        # Draw P term
        cr.set_source_rgb(0.2, 0.6, 0.2)
        cr.set_line_width(2)
        cr.new_path()
        first_point = True
        for t, val in zip(times, p_vals):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        
        # Draw I term
        cr.set_source_rgb(0.8, 0.8, 0.2)
        cr.set_line_width(2)
        cr.new_path()
        first_point = True
        for t, val in zip(times, i_vals):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        
        # Draw D term
        cr.set_source_rgb(0.2, 0.6, 0.6)
        cr.set_line_width(2)
        cr.new_path()
        first_point = True
        for t, val in zip(times, d_vals):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        
        # Draw FF term
        cr.set_source_rgb(0.6, 0.2, 0.8)
        cr.set_line_width(2)
        cr.new_path()
        first_point = True
        for t, val in zip(times, ff_vals):
            x = calc_x(t)
            y = calc_y(val)
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        cr.stroke()
        
        # Legend
        cr.set_font_size(10)
        legend_y = 20
        legend_x = width - 140
        
        cr.set_source_rgb(0.2, 0.6, 0.2)
        cr.move_to(legend_x, legend_y)
        cr.show_text("─ P term")
        
        cr.set_source_rgb(0.8, 0.8, 0.2)
        cr.move_to(legend_x, legend_y + 15)
        cr.show_text("─ I term")
        
        cr.set_source_rgb(0.2, 0.6, 0.6)
        cr.move_to(legend_x, legend_y + 30)
        cr.show_text("─ D term")
        
        cr.set_source_rgb(0.6, 0.2, 0.8)
        cr.move_to(legend_x, legend_y + 45)
        cr.show_text("─ FF term")


class PIDFFSimulatorApp(Adw.Application):
    """Main application class."""
    
    def __init__(self):
        super().__init__(application_id="org.ardupilot.pidffsimulator")
        
    def do_activate(self):
        """Called when application is activated."""
        window = PIDFFSimulatorWindow(self)
        window.present()


def main():
    """Main entry point."""
    app = PIDFFSimulatorApp()
    app.run(None)


if __name__ == "__main__":
    main()
