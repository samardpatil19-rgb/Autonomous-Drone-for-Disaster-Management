"""
GPS Reporter Module
====================
Logs person detections with GPS coordinates and timestamps.
Reports can be saved to CSV and printed to console.
"""

import os
import csv
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GPSReporter:
    """Logs and reports person detections with GPS coordinates."""

    def __init__(self, log_file="detection_log.csv", enable_csv=True,
                 enable_console=True, save_images=False, images_dir="detections/"):
        """
        Initialize GPS reporter.

        Args:
            log_file: Path to CSV log file
            enable_csv: Whether to log to CSV
            enable_console: Whether to print alerts to console
            save_images: Whether to save detection frames
            images_dir: Directory for saved detection images
        """
        self.log_file = log_file
        self.enable_csv = enable_csv
        self.enable_console = enable_console
        self.save_images = save_images
        self.images_dir = images_dir
        self.detection_count = 0

        # Initialize CSV file
        if self.enable_csv:
            self._init_csv()

        # Create images directory
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

    def _init_csv(self):
        """Initialize CSV log file with headers."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "detection_id",
                    "timestamp",
                    "latitude",
                    "longitude",
                    "altitude",
                    "num_persons",
                    "max_confidence",
                    "heading",
                    "groundspeed",
                    "battery_level",
                ])
            logger.info(f"Created detection log: {self.log_file}")

    def report_detection(self, gps_data, detections, drone_info=None, frame=None):
        """
        Report a person detection event.

        Args:
            gps_data: dict with 'lat', 'lon', 'alt' keys
            detections: list of detection dicts from PersonDetector
            drone_info: optional dict with 'heading', 'groundspeed', 'battery_level'
            frame: optional numpy array (frame to save as image)
        """
        self.detection_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_persons = len(detections)
        max_confidence = max(d["confidence"] for d in detections) if detections else 0

        # Console alert
        if self.enable_console:
            self._print_alert(timestamp, gps_data, num_persons, max_confidence)

        # CSV logging
        if self.enable_csv and gps_data:
            self._log_csv(timestamp, gps_data, num_persons, max_confidence, drone_info)

        # Save detection image
        if self.save_images and frame is not None:
            self._save_image(frame, timestamp)

        return self.detection_count

    def _print_alert(self, timestamp, gps_data, num_persons, max_confidence):
        """Print a detection alert to console."""
        print()
        print("🚨" + "=" * 56 + "🚨")
        print(f"  🔴 PERSON DETECTED! ({num_persons} person(s))")
        print(f"  📍 Coordinates:")
        if gps_data:
            print(f"       Latitude:  {gps_data['lat']:.6f}")
            print(f"       Longitude: {gps_data['lon']:.6f}")
            print(f"       Altitude:  {gps_data['alt']:.1f} m")
        else:
            print(f"       GPS: Not available")
        print(f"  🎯 Confidence: {max_confidence:.1%}")
        print(f"  🕐 Time: {timestamp}")
        print(f"  #️⃣  Detection #{self.detection_count}")
        print("🚨" + "=" * 56 + "🚨")
        print()

    def _log_csv(self, timestamp, gps_data, num_persons, max_confidence, drone_info):
        """Append detection to CSV log."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.detection_count,
                    timestamp,
                    f"{gps_data['lat']:.6f}" if gps_data else "",
                    f"{gps_data['lon']:.6f}" if gps_data else "",
                    f"{gps_data['alt']:.1f}" if gps_data else "",
                    num_persons,
                    f"{max_confidence:.3f}",
                    drone_info.get("heading", "") if drone_info else "",
                    drone_info.get("groundspeed", "") if drone_info else "",
                    drone_info.get("battery_level", "") if drone_info else "",
                ])
            logger.debug(f"Detection #{self.detection_count} logged to CSV")
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")

    def _save_image(self, frame, timestamp):
        """Save detection frame as image."""
        try:
            import cv2
            filename = os.path.join(
                self.images_dir,
                f"detection_{self.detection_count}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
            )
            cv2.imwrite(filename, frame)
            logger.debug(f"Detection image saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save detection image: {e}")

    def get_summary(self):
        """Get summary of all detections."""
        return {
            "total_detections": self.detection_count,
            "log_file": self.log_file,
        }
