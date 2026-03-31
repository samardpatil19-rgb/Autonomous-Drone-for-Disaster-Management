"""
Dual-Channel Communications Module
====================================
Handles sending detection alerts through two independent channels:

Channel 1 (Telemetry): MAVLink STATUSTEXT via Pixhawk telemetry radio.
    - Works without internet, even in remote wilderness.
    - Appears as a text alert inside Mission Planner on the laptop.

Channel 2 (LTE/ZeroTier): HTTP POST with GPS + screenshot.
    - Sends data over the 4G dongle via ZeroTier VPN.
    - Delivers the actual detection image to the ground station.
"""

import os
import time
import logging
import threading

logger = logging.getLogger(__name__)


class Communications:
    """Dual-channel detection alert system."""

    def __init__(self, ground_station_url=None, enable_mavlink=True,
                 enable_http=True, save_local=True, detections_dir="detections/"):
        """
        Initialize the communications module.

        Args:
            ground_station_url: URL of the ground station HTTP endpoint
                                (e.g., http://10.147.x.x:8080/alert)
            enable_mavlink: Send MAVLink STATUSTEXT alerts via Pixhawk
            enable_http: Send HTTP POST alerts via 4G/ZeroTier
            save_local: Save detection images locally to SD card
            detections_dir: Directory for local detection images
        """
        self.ground_station_url = ground_station_url
        self.enable_mavlink = enable_mavlink
        self.enable_http = enable_http
        self.save_local = save_local
        self.detections_dir = detections_dir
        self.detection_count = 0

        # Create detections directory
        if self.save_local:
            os.makedirs(self.detections_dir, exist_ok=True)

        logger.info("Communications initialized:")
        logger.info(f"  MAVLink alerts: {'ON' if enable_mavlink else 'OFF'}")
        logger.info(f"  HTTP alerts:    {'ON' if enable_http else 'OFF'}")
        logger.info(f"  Local save:     {'ON' if save_local else 'OFF'}")

    def send_alert(self, gps_data, detections, drone_controller=None,
                   frame=None, drone_info=None):
        """
        Send a detection alert through all enabled channels.

        Args:
            gps_data: dict with 'lat', 'lon', 'alt'
            detections: list of detection dicts from PersonDetector
            drone_controller: DroneController instance (for MAVLink)
            frame: numpy array (annotated frame with bounding boxes)
            drone_info: optional dict with heading, speed, battery
        """
        self.detection_count += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        num_persons = len(detections)
        max_conf = max(d["confidence"] for d in detections) if detections else 0

        # ── Console Alert (always) ──
        self._console_alert(timestamp, gps_data, num_persons, max_conf)

        # ── Channel 1: MAVLink Telemetry ──
        if self.enable_mavlink and drone_controller:
            self._send_mavlink_alert(drone_controller, gps_data, num_persons)

        # ── Channel 2: HTTP over LTE/ZeroTier ──
        if self.enable_http and self.ground_station_url:
            # Send HTTP in a background thread so it doesn't block detection
            threading.Thread(
                target=self._send_http_alert,
                args=(gps_data, num_persons, max_conf, timestamp, frame),
                daemon=True,
            ).start()

        # ── Save to SD card ──
        if self.save_local and frame is not None:
            self._save_detection_image(frame, timestamp)

        # ── Log to CSV ──
        self._log_csv(timestamp, gps_data, num_persons, max_conf, drone_info)

        return self.detection_count

    def _console_alert(self, timestamp, gps_data, num_persons, max_conf):
        """Print a bold detection alert to the Pi's console."""
        print()
        print("=" * 60)
        print(f"  PERSON DETECTED! ({num_persons} person(s))")
        print(f"  Coordinates:")
        if gps_data:
            print(f"       Latitude:  {gps_data['lat']:.6f}")
            print(f"       Longitude: {gps_data['lon']:.6f}")
            print(f"       Altitude:  {gps_data['alt']:.1f} m")
        else:
            print(f"       GPS: Not available")
        print(f"  Confidence: {max_conf:.1%}")
        print(f"  Time: {timestamp}")
        print(f"  Detection #{self.detection_count}")
        print("=" * 60)
        print()

    def _send_mavlink_alert(self, drone_controller, gps_data, num_persons):
        """
        Send a STATUSTEXT message through the Pixhawk telemetry radio.
        This appears as a red text alert in Mission Planner.
        """
        try:
            if drone_controller.vehicle is None:
                logger.warning("Cannot send MAVLink alert: no vehicle connected")
                return

            from pymavlink import mavutil

            # Build the alert message (max 50 chars for MAVLink STATUSTEXT)
            if gps_data:
                msg = f"PERSON! {gps_data['lat']:.5f},{gps_data['lon']:.5f} x{num_persons}"
            else:
                msg = f"PERSON DETECTED! Count: {num_persons}"

            # Send through Pixhawk's MAVLink connection
            drone_controller.vehicle.message_factory
            master = drone_controller.vehicle._master

            master.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_CRITICAL,
                msg.encode('utf-8')
            )

            logger.info(f"MAVLink STATUSTEXT sent: {msg}")

        except Exception as e:
            logger.error(f"Failed to send MAVLink alert: {e}")

    def _send_http_alert(self, gps_data, num_persons, max_conf, timestamp, frame):
        """
        Send an HTTP POST request with detection data to the ground station.
        Runs in a background thread to avoid blocking the detection loop.
        """
        try:
            import requests
            import json
            import base64
            import cv2

            payload = {
                "timestamp": timestamp,
                "detection_id": self.detection_count,
                "num_persons": num_persons,
                "max_confidence": max_conf,
                "gps": gps_data,
            }

            # Attach the detection image as base64 if available
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                payload["image_base64"] = base64.b64encode(buffer).decode('utf-8')

            response = requests.post(
                self.ground_station_url,
                json=payload,
                timeout=5,
            )

            if response.status_code == 200:
                logger.info(f"HTTP alert sent to ground station")
            else:
                logger.warning(f"Ground station returned status {response.status_code}")

        except Exception as e:
            logger.warning(f"HTTP alert failed (ground station may be offline): {e}")

    def _save_detection_image(self, frame, timestamp):
        """Save the annotated frame to the SD card."""
        try:
            import cv2
            safe_ts = timestamp.replace(":", "-").replace(" ", "_")
            filename = os.path.join(
                self.detections_dir,
                f"detection_{self.detection_count}_{safe_ts}.jpg"
            )
            cv2.imwrite(filename, frame)
            logger.info(f"Detection image saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save detection image: {e}")

    def _log_csv(self, timestamp, gps_data, num_persons, max_conf, drone_info):
        """Append detection to the CSV log file."""
        try:
            import csv
            log_file = os.path.join(self.detections_dir, "detection_log.csv")
            file_exists = os.path.exists(log_file)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "detection_id", "timestamp", "latitude", "longitude",
                        "altitude", "num_persons", "max_confidence",
                        "heading", "groundspeed", "battery"
                    ])
                writer.writerow([
                    self.detection_count,
                    timestamp,
                    f"{gps_data['lat']:.6f}" if gps_data else "",
                    f"{gps_data['lon']:.6f}" if gps_data else "",
                    f"{gps_data['alt']:.1f}" if gps_data else "",
                    num_persons,
                    f"{max_conf:.3f}",
                    drone_info.get("heading", "") if drone_info else "",
                    drone_info.get("groundspeed", "") if drone_info else "",
                    drone_info.get("battery_level", "") if drone_info else "",
                ])
            logger.debug(f"Detection #{self.detection_count} logged to CSV")
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")

    def get_summary(self):
        """Return a summary of all detections."""
        return {
            "total_detections": self.detection_count,
            "detections_dir": self.detections_dir,
        }
