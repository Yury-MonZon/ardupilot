#!/usr/bin/env python3
"""
Log Parser for ArduPilot binary logs
Extracts PID, ATRP, MSG, MODE, PARM, and AETR (speed scaler) messages
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os

try:
    from pymavlink import mavutil
except ImportError:
    print("Error: pymavlink not found. Install with: pip install pymavlink")
    raise


@dataclass
class PIDData:
    """Container for PID message data (PIDR/PIDP)"""

    time: np.ndarray = field(default_factory=lambda: np.array([]))
    target: np.ndarray = field(default_factory=lambda: np.array([]))
    actual: np.ndarray = field(default_factory=lambda: np.array([]))
    error: np.ndarray = field(default_factory=lambda: np.array([]))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    I: np.ndarray = field(default_factory=lambda: np.array([]))
    D: np.ndarray = field(default_factory=lambda: np.array([]))
    FF: np.ndarray = field(default_factory=lambda: np.array([]))
    DFF: np.ndarray = field(default_factory=lambda: np.array([]))
    Dmod: np.ndarray = field(default_factory=lambda: np.array([]))
    SRate: np.ndarray = field(default_factory=lambda: np.array([]))
    Flags: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rate: float = 0.0

    def __len__(self):
        return len(self.time)

    def is_empty(self):
        return len(self.time) == 0


@dataclass
class ATRPData:
    """Container for ATRP (autotune) message data"""

    time: np.ndarray = field(default_factory=lambda: np.array([]))
    axis: np.ndarray = field(default_factory=lambda: np.array([]))
    state: np.ndarray = field(default_factory=lambda: np.array([]))
    actuator: np.ndarray = field(default_factory=lambda: np.array([]))
    P_slew: np.ndarray = field(default_factory=lambda: np.array([]))
    D_slew: np.ndarray = field(default_factory=lambda: np.array([]))
    FF_single: np.ndarray = field(default_factory=lambda: np.array([]))
    FF: np.ndarray = field(default_factory=lambda: np.array([]))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    I: np.ndarray = field(default_factory=lambda: np.array([]))
    D: np.ndarray = field(default_factory=lambda: np.array([]))
    action: np.ndarray = field(default_factory=lambda: np.array([]))
    rmax: np.ndarray = field(default_factory=lambda: np.array([]))
    tau: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SpeedScalerData:
    """Container for speed scaler data (from AETR.SS)"""

    time: np.ndarray = field(default_factory=lambda: np.array([]))
    scaler: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AltitudeSample:
    """Single altitude sample"""

    time_s: float
    altitude_m: float


@dataclass
class BatterySample:
    """Single battery measurement"""

    time_s: float
    voltage: float
    current: float


@dataclass
class GPSSample:
    """Single GPS measurement"""

    time_s: float
    speed_mps: float
    climb_rate_mps: float
    ground_course_deg: float = 0.0


@dataclass
class BAROSample:
    """Single barometer measurement with climb rate"""

    time_s: float
    altitude_m: float
    climb_rate_mps: float


@dataclass
class AirspeedSample:
    """Single airspeed measurement (true airspeed, wind-agnostic)"""

    time_s: float
    airspeed_mps: float


@dataclass
class ModeEvent:
    """Flight mode change event"""

    time_us: int
    mode: int
    mode_num: int
    reason: int


@dataclass
class AutotuneEvent:
    """Autotune start/stop event from MSG"""

    time_s: float
    event_type: str
    message: str


@dataclass
class LogData:
    """Container for all parsed log data"""

    roll: Optional[PIDData] = None
    pitch: Optional[PIDData] = None
    atrp: Optional[ATRPData] = None
    speed_scaler: Optional[SpeedScalerData] = None
    altitude: List[AltitudeSample] = field(default_factory=list)
    battery: List[BatterySample] = field(default_factory=list)
    gps: List[GPSSample] = field(default_factory=list)
    baro: List[BAROSample] = field(default_factory=list)
    airspeed: List[AirspeedSample] = field(default_factory=list)
    modes: List[ModeEvent] = field(default_factory=list)
    autotune_events: List[AutotuneEvent] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    filename: str = ""


class LogParser:
    """Parse ArduPilot binary logs"""

    MODE_NAMES = {
        0: "MANUAL",
        1: "CIRCLE",
        2: "STABILIZE",
        3: "TRAINING",
        4: "ACRO",
        5: "FBWA",
        6: "FBWB",
        7: "CRUISE",
        8: "AUTOTUNE",
        9: "UNUSED_9",
        10: "AUTO",
        11: "RTL",
        12: "LOITER",
        13: "TAKEOFF",
        14: "AVOID_ADSB",
        15: "GUIDED",
        16: "INITIALISING",
    }

    # Modes suitable for PID analysis (FBWA, FBWB, CRUISE, AUTO, RTL, LOITER, GUIDED)
    SUITABLE_MODES = {4, 5, 6, 7, 10, 11, 12, 15}
    AUTOTUNE_MODE = 8
    # Modes to always skip (MANUAL, TAKEOFF, INITIALISING)
    SKIP_MODES = {0, 13, 16}

    def __init__(self, logfile: str):
        self.logfile = logfile
        if not os.path.exists(logfile):
            raise FileNotFoundError(f"Log file not found: {logfile}")

    def parse(self) -> LogData:
        """Parse the log file and return structured data"""
        print(f"Parsing: {self.logfile}")

        mlog = mavutil.mavlink_connection(self.logfile)

        # Initialize data containers
        roll_data = {
            "time": [],
            "target": [],
            "actual": [],
            "error": [],
            "P": [],
            "I": [],
            "D": [],
            "FF": [],
            "DFF": [],
            "Dmod": [],
            "SRate": [],
            "Flags": [],
        }
        pitch_data = {
            "time": [],
            "target": [],
            "actual": [],
            "error": [],
            "P": [],
            "I": [],
            "D": [],
            "FF": [],
            "DFF": [],
            "Dmod": [],
            "SRate": [],
            "Flags": [],
        }
        atrp_data = {
            "time": [],
            "axis": [],
            "state": [],
            "actuator": [],
            "P_slew": [],
            "D_slew": [],
            "FF_single": [],
            "FF": [],
            "P": [],
            "I": [],
            "D": [],
            "action": [],
            "rmax": [],
            "tau": [],
        }
        scaler_data = {"time": [], "scaler": []}
        altitude_data: List[AltitudeSample] = []
        battery_data: List[BatterySample] = []
        gps_data: List[GPSSample] = []
        baro_data: List[BAROSample] = []
        airspeed_data: List[AirspeedSample] = []
        modes: List[ModeEvent] = []
        autotune_events: List[AutotuneEvent] = []
        params: Dict[str, float] = {}

        first_time = None
        last_time = None

        msg_count = 0
        while True:
            msg = mlog.recv_match()
            if msg is None:
                break

            msg_count += 1
            if msg_count % 50000 == 0:
                print(f"  Processed {msg_count} messages...")

            msg_type = msg.get_type()
            time_us = getattr(msg, "TimeUS", None)

            # Track duration from messages that have valid timestamps
            # FMT and other system messages may not have TimeUS
            if time_us is not None and time_us > 0:
                if first_time is None:
                    first_time = time_us
                last_time = time_us

            # Parse PIDR (Roll rate PID)
            if msg_type == "PIDR":
                roll_data["time"].append(time_us * 1e-6)
                roll_data["target"].append(getattr(msg, "Tar", 0))
                roll_data["actual"].append(getattr(msg, "Act", 0))
                roll_data["error"].append(getattr(msg, "Err", 0))
                roll_data["P"].append(getattr(msg, "P", 0))
                roll_data["I"].append(getattr(msg, "I", 0))
                roll_data["D"].append(getattr(msg, "D", 0))
                roll_data["FF"].append(getattr(msg, "FF", 0))
                roll_data["DFF"].append(getattr(msg, "DFF", 0))
                roll_data["Dmod"].append(getattr(msg, "Dmod", 1.0))
                roll_data["SRate"].append(getattr(msg, "SRate", 0))
                roll_data["Flags"].append(getattr(msg, "Flags", 0))

            # Parse PIDP (Pitch rate PID)
            elif msg_type == "PIDP":
                pitch_data["time"].append(time_us * 1e-6)
                pitch_data["target"].append(getattr(msg, "Tar", 0))
                pitch_data["actual"].append(getattr(msg, "Act", 0))
                pitch_data["error"].append(getattr(msg, "Err", 0))
                pitch_data["P"].append(getattr(msg, "P", 0))
                pitch_data["I"].append(getattr(msg, "I", 0))
                pitch_data["D"].append(getattr(msg, "D", 0))
                pitch_data["FF"].append(getattr(msg, "FF", 0))
                pitch_data["DFF"].append(getattr(msg, "DFF", 0))
                pitch_data["Dmod"].append(getattr(msg, "Dmod", 1.0))
                pitch_data["SRate"].append(getattr(msg, "SRate", 0))
                pitch_data["Flags"].append(getattr(msg, "Flags", 0))

            # Parse ATRP (Autotune data)
            elif msg_type == "ATRP":
                atrp_data["time"].append(time_us * 1e-6)
                atrp_data["axis"].append(getattr(msg, "Axis", 0))
                atrp_data["state"].append(getattr(msg, "State", 0))
                atrp_data["actuator"].append(getattr(msg, "Sur", 0))
                atrp_data["P_slew"].append(getattr(msg, "PSlew", 0))
                atrp_data["D_slew"].append(getattr(msg, "DSlew", 0))
                atrp_data["FF_single"].append(getattr(msg, "FF0", 0))
                atrp_data["FF"].append(getattr(msg, "FF", 0))
                atrp_data["P"].append(getattr(msg, "P", 0))
                atrp_data["I"].append(getattr(msg, "I", 0))
                atrp_data["D"].append(getattr(msg, "D", 0))
                atrp_data["action"].append(getattr(msg, "Action", 0))
                atrp_data["rmax"].append(getattr(msg, "RMAX", 0))
                atrp_data["tau"].append(getattr(msg, "TAU", 0))

            # Parse AETR for speed scaler (SS field)
            elif msg_type == "AETR":
                ss = getattr(msg, "SS", 1.0)
                scaler_data["time"].append(time_us * 1e-6)
                scaler_data["scaler"].append(ss)

            # Parse MODE (flight mode changes)
            elif msg_type == "MODE":
                modes.append(
                    ModeEvent(
                        time_us=time_us,
                        mode=getattr(msg, "Mode", 0),
                        mode_num=getattr(msg, "ModeNum", 0),
                        reason=getattr(msg, "Rsn", 0),
                    )
                )

            # Parse MSG (for autotune events)
            elif msg_type == "MSG":
                message = getattr(msg, "Message", "")
                time_s = time_us * 1e-6

                if "Started autotune" in message:
                    autotune_events.append(
                        AutotuneEvent(
                            time_s=time_s, event_type="start", message=message
                        )
                    )
                elif "Stopped autotune" in message:
                    autotune_events.append(
                        AutotuneEvent(time_s=time_s, event_type="stop", message=message)
                    )
                elif "Roll: Finished" in message:
                    autotune_events.append(
                        AutotuneEvent(
                            time_s=time_s, event_type="roll_finished", message=message
                        )
                    )
                elif "Pitch: Finished" in message:
                    autotune_events.append(
                        AutotuneEvent(
                            time_s=time_s, event_type="pitch_finished", message=message
                        )
                    )

            # Parse BARO for altitude and climb rate
            elif msg_type == "BARO":
                alt = getattr(msg, "Alt", 0)
                crt = getattr(msg, "CRt", 0)
                altitude_data.append(
                    AltitudeSample(time_s=time_us * 1e-6, altitude_m=alt)
                )
                baro_data.append(
                    BAROSample(
                        time_s=time_us * 1e-6, altitude_m=alt, climb_rate_mps=crt
                    )
                )

            # Parse BAT (battery)
            elif msg_type == "BAT":
                battery_data.append(
                    BatterySample(
                        time_s=time_us * 1e-6,
                        voltage=getattr(msg, "Volt", 0),
                        current=getattr(msg, "Curr", 0),
                    )
                )

            # Parse GPS (for speed and climb rate)
            elif msg_type == "GPS":
                gps_data.append(
                    GPSSample(
                        time_s=time_us * 1e-6,
                        speed_mps=getattr(msg, "Spd", 0),
                        climb_rate_mps=getattr(msg, "VZ", 0),
                        ground_course_deg=getattr(msg, "GCrs", 0),
                    )
                )

            # Parse ARSP (airspeed - true airspeed, wind-agnostic)
            elif msg_type == "ARSP":
                airspeed_data.append(
                    AirspeedSample(
                        time_s=time_us * 1e-6,
                        airspeed_mps=getattr(msg, "Airspeed", 0),
                    )
                )

            # Parse PARM (parameters)
            elif msg_type == "PARM":
                name = getattr(msg, "Name", "")
                value = getattr(msg, "Value", 0)
                if name:
                    params[name] = value

        print(f"  Total messages: {msg_count}")

        # Convert to numpy arrays
        roll_pid = self._dict_to_pid_data(roll_data)
        pitch_pid = self._dict_to_pid_data(pitch_data)

        # Convert ATRP data
        atrp = None
        if atrp_data["time"]:
            atrp = ATRPData(
                time=np.array(atrp_data["time"]),
                axis=np.array(atrp_data["axis"]),
                state=np.array(atrp_data["state"]),
                actuator=np.array(atrp_data["actuator"]),
                P_slew=np.array(atrp_data["P_slew"]),
                D_slew=np.array(atrp_data["D_slew"]),
                FF_single=np.array(atrp_data["FF_single"]),
                FF=np.array(atrp_data["FF"]),
                P=np.array(atrp_data["P"]),
                I=np.array(atrp_data["I"]),
                D=np.array(atrp_data["D"]),
                action=np.array(atrp_data["action"]),
                rmax=np.array(atrp_data["rmax"]),
                tau=np.array(atrp_data["tau"]),
            )

        # Convert speed scaler data
        speed_scaler = None
        if scaler_data["time"]:
            speed_scaler = SpeedScalerData(
                time=np.array(scaler_data["time"]),
                scaler=np.array(scaler_data["scaler"]),
            )

        duration = 0.0
        start_time_s = 0.0
        end_time_s = 0.0
        if first_time and last_time:
            duration = (last_time - first_time) * 1e-6
            start_time_s = first_time * 1e-6
            end_time_s = last_time * 1e-6

        return LogData(
            roll=roll_pid,
            pitch=pitch_pid,
            atrp=atrp,
            speed_scaler=speed_scaler,
            altitude=altitude_data,
            battery=battery_data,
            gps=gps_data,
            baro=baro_data,
            airspeed=airspeed_data,
            modes=modes,
            autotune_events=autotune_events,
            params=params,
            duration_s=duration,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            filename=os.path.basename(self.logfile),
        )

    def _dict_to_pid_data(self, data_dict: dict) -> PIDData:
        """Convert dictionary to PIDData with numpy arrays"""
        if not data_dict["time"]:
            return PIDData()

        time_arr = np.array(data_dict["time"])

        if len(time_arr) > 1:
            dt = np.diff(time_arr)
            sample_rate = 1.0 / np.median(dt) if np.median(dt) > 0 else 0.0
        else:
            sample_rate = 0.0

        return PIDData(
            time=time_arr,
            target=np.array(data_dict["target"]),
            actual=np.array(data_dict["actual"]),
            error=np.array(data_dict["error"]),
            P=np.array(data_dict["P"]),
            I=np.array(data_dict["I"]),
            D=np.array(data_dict["D"]),
            FF=np.array(data_dict["FF"]),
            DFF=np.array(data_dict["DFF"]),
            Dmod=np.array(data_dict["Dmod"]),
            SRate=np.array(data_dict["SRate"]),
            Flags=np.array(data_dict["Flags"]),
            sample_rate=sample_rate,
        )

    @staticmethod
    def get_mode_name(mode_num: int) -> str:
        """Get flight mode name from number"""
        return LogParser.MODE_NAMES.get(mode_num, f"UNKNOWN_{mode_num}")

    @staticmethod
    def is_suitable_mode(mode_num: int) -> bool:
        """Check if mode is suitable for PID analysis"""
        return mode_num in LogParser.SUITABLE_MODES

    @staticmethod
    def should_skip_mode(mode_num: int) -> bool:
        """Check if mode should always be skipped (MANUAL, TAKEOFF, or UNKNOWN)"""
        # Skip MANUAL (0), TAKEOFF (12), and any negative or very high mode numbers (UNKNOWN)
        if mode_num in LogParser.SKIP_MODES:
            return True
        if mode_num < 0 or mode_num > 30:
            return True
        return False

    @staticmethod
    def is_autotune_mode(mode_num: int) -> bool:
        """Check if mode is autotune"""
        return mode_num == LogParser.AUTOTUNE_MODE


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <logfile.bin>")
        sys.exit(1)

    parser = LogParser(sys.argv[1])
    data = parser.parse()

    print(f"\nLog Summary:")
    print(f"  Duration: {data.duration_s:.1f} seconds")
    print(f"  Roll samples: {len(data.roll.time) if data.roll else 0}")
    print(f"  Pitch samples: {len(data.pitch.time) if data.pitch else 0}")
    print(
        f"  Speed scaler samples: {len(data.speed_scaler.time) if data.speed_scaler else 0}"
    )
    print(f"  Altitude samples: {len(data.altitude)}")
    print(f"  Mode changes: {len(data.modes)}")
    print(f"  Autotune events: {len(data.autotune_events)}")
    print(f"  Parameters: {len(data.params)}")
