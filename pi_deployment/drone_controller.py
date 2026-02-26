"""
Drone Controller Module
========================
DroneKit-based interface for Pixhawk 2.4.8 flight controller.
Handles connection, mode switching, and GPS retrieval.
"""

import time
import logging

logger = logging.getLogger(__name__)


class DroneController:
    """Interface to Pixhawk 2.4.8 via DroneKit/MAVLink."""

    def __init__(self, connection_string, baud_rate=57600, timeout=30):
        """
        Initialize drone connection.

        Args:
            connection_string: Serial port (e.g., '/dev/serial0')
            baud_rate: MAVLink baud rate (default 57600)
            timeout: Connection timeout in seconds
        """
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.vehicle = None

    def connect(self):
        """Establish connection to Pixhawk."""
        try:
            from dronekit import connect as dk_connect

            logger.info(f"Connecting to Pixhawk at {self.connection_string} "
                        f"(baud={self.baud_rate})...")

            self.vehicle = dk_connect(
                self.connection_string,
                baud=self.baud_rate,
                wait_ready=True,
                heartbeat_timeout=self.timeout,
            )

            logger.info("Connected to Pixhawk!")
            logger.info(f"  Firmware: {self.vehicle.version}")
            logger.info(f"  GPS: {self.vehicle.gps_0}")
            logger.info(f"  Battery: {self.vehicle.battery}")
            logger.info(f"  Mode: {self.vehicle.mode.name}")
            logger.info(f"  Armed: {self.vehicle.armed}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Pixhawk: {e}")
            return False

    def hover(self):
        """
        Switch to LOITER mode to hover in place.
        The drone will maintain its current position and altitude.
        """
        if not self.vehicle:
            logger.error("Not connected to drone!")
            return False

        try:
            from dronekit import VehicleMode

            logger.info("🛑 Switching to LOITER mode (hover in place)...")
            self.vehicle.mode = VehicleMode("LOITER")

            # Wait for mode change to take effect
            timeout = 5
            start = time.time()
            while self.vehicle.mode.name != "LOITER":
                if time.time() - start > timeout:
                    logger.warning("Mode change timeout — may not have switched to LOITER")
                    return False
                time.sleep(0.5)

            logger.info("✅ Drone is now hovering (LOITER mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to switch to LOITER: {e}")
            return False

    def resume_mission(self):
        """
        Switch back to AUTO mode to continue the waypoint mission.
        """
        if not self.vehicle:
            logger.error("Not connected to drone!")
            return False

        try:
            from dronekit import VehicleMode

            logger.info("▶️ Resuming mission (switching to AUTO mode)...")
            self.vehicle.mode = VehicleMode("AUTO")

            # Wait for mode change
            timeout = 5
            start = time.time()
            while self.vehicle.mode.name != "AUTO":
                if time.time() - start > timeout:
                    logger.warning("Mode change timeout — may not have switched to AUTO")
                    return False
                time.sleep(0.5)

            logger.info("✅ Mission resumed (AUTO mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to resume mission: {e}")
            return False

    def get_gps(self):
        """
        Get current GPS coordinates.

        Returns:
            dict: {'lat': float, 'lon': float, 'alt': float} or None
        """
        if not self.vehicle:
            logger.error("Not connected to drone!")
            return None

        try:
            location = self.vehicle.location.global_relative_frame
            return {
                "lat": location.lat,
                "lon": location.lon,
                "alt": location.alt,
            }
        except Exception as e:
            logger.error(f"Failed to get GPS: {e}")
            return None

    def get_mode(self):
        """Get current flight mode."""
        if self.vehicle:
            return self.vehicle.mode.name
        return "UNKNOWN"

    def is_armed(self):
        """Check if the drone is armed."""
        if self.vehicle:
            return self.vehicle.armed
        return False

    def get_battery(self):
        """Get battery status."""
        if self.vehicle:
            return {
                "voltage": self.vehicle.battery.voltage,
                "current": self.vehicle.battery.current,
                "level": self.vehicle.battery.level,
            }
        return None

    def get_heading(self):
        """Get current heading in degrees."""
        if self.vehicle:
            return self.vehicle.heading
        return None

    def get_groundspeed(self):
        """Get current ground speed in m/s."""
        if self.vehicle:
            return self.vehicle.groundspeed
        return None

    def disconnect(self):
        """Close connection to Pixhawk."""
        if self.vehicle:
            logger.info("Disconnecting from Pixhawk...")
            self.vehicle.close()
            self.vehicle = None
            logger.info("Disconnected.")
