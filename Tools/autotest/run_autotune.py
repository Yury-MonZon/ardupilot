from pymavlink import mavutil
import time

        # MANUAL        = 0,
        # CIRCLE        = 1,
        # STABILIZE     = 2,
        # TRAINING      = 3,
        # ACRO          = 4,
        # FLY_BY_WIRE_A = 5,
        # FLY_BY_WIRE_B = 6,
        # CRUISE        = 7,
        # AUTOTUNE      = 8,
        # AUTO          = 10,
        # RTL           = 11,
        # LOITER        = 12,
        # TAKEOFF       = 13,
        # AVOID_ADSB    = 14,
        # GUIDED        = 15,
        # INITIALISING  = 16,
        # THERMAL       = 24,

# Function to simulate RC channel overrides
def send_rc_override(roll, pitch, yaw):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        roll,   # roll (Channel 1)
        pitch,  # pitch (Channel 2)
        1500,   # throttle (Channel 3)
        yaw,   # yaw (Channel 4)
        0, 0, 0, 0)

def send_roll(roll):
    send_rc_override(roll, 1500, 1500)


def send_pitch(pitch):
    send_rc_override(1500, pitch, 1500)


def send_yaw(yaw):
    send_rc_override(1500, 1500, yaw)


# Connect to SITL
master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')

# Wait for a heartbeat before sending commands
master.wait_heartbeat()

# Switch to AutoTune mode
master.set_mode(8)  # 8 is the code for AutoTune mode
    
# Simulate stick movements for AutoTune
for a in range(20):
    # Simulate roll movement
    print(f"Roll {a}")
    send_roll(1900)  # Roll right
    time.sleep(1)
    send_roll(1500)  # Roll left
    time.sleep(1)
    send_roll(1100)  # Roll left
    time.sleep(1)
    send_roll(1500)  # Roll left
    time.sleep(1)

for a in range(20):
    # Simulate pitch movement
    print(f"Pitch {a}")
    send_pitch(1900)  # Pitch up
    time.sleep(1)
    send_pitch(1500)  # Pitch up
    time.sleep(1)
    send_pitch(1100)  # Pitch down
    time.sleep(1)
    send_pitch(1500)  # Pitch up
    time.sleep(1)

for a in range(20):
    # Simulate pitch movement
    print(f"Yaw {a}")
    send_yaw(1900)  # Pitch up
    time.sleep(1)
    send_yaw(1500)  # Pitch up
    time.sleep(1)
    send_yaw(1100)  # Pitch down
    time.sleep(1)
    send_yaw(1500)  # Pitch up
    time.sleep(1)

# Switch back to stabilize mode after tuning
master.set_mode(11) 
