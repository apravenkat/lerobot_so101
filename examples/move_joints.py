from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig

# --- Configure the robot (update the serial port to your own, e.g. /dev/ttyUSB0 or COM3)
config = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True)

# --- Create and connect the robot
arm = SO101Follower(config)
arm.connect(calibrate=False)

# --- Get the current joint angles
print(arm.is_calibrated)
observation = arm.get_observation()
print("Joint Angles:")
for name, angle in observation.items():
    if name.endswith(".pos"):
        print(f"{name}: {angle:.2f}")

# --- Disconnect safely when done
arm.disconnect()
