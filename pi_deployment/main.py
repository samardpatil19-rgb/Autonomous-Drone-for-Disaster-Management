"""
Main Entry Point — Drone Person Detection System
==================================================
This is the main script that runs on the Raspberry Pi 5.
It combines the detector, drone controller, and GPS reporter
into a single detection loop.

Flow:
    1. Connect to Pixhawk 2.4.8
    2. Initialize camera and ML model
    3. Capture frames → detect persons → hover → report GPS
    4. Resume mission after hover duration

Usage:
    python main.py                  # Normal mode (with drone)
    python main.py --test-camera    # Camera test only (no drone)
    python main.py --simulate       # Simulate mode (no hardware)
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Import modules
from config import (
    PIXHAWK_CONNECTION, PIXHAWK_BAUD, PIXHAWK_TIMEOUT,
    MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_CLASS,
    DETECTION_INTERVAL, HOVER_DURATION, COOLDOWN_AFTER_DETECTION,
    MIN_DETECTION_COUNT, GPS_LOG_FILE, ENABLE_CONSOLE_ALERTS,
    ENABLE_CSV_LOGGING, SAVE_DETECTION_IMAGES, DETECTION_IMAGES_DIR,
    CAMERA_RESOLUTION, USE_PI_CAMERA, USB_CAMERA_INDEX,
    DISPLAY_PREVIEW, LOG_LEVEL,
)
from detector import PersonDetector
from drone_controller import DroneController
from gps_reporter import GPSReporter


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_camera():
    """Initialize camera (Pi Camera or USB webcam)."""
    logger = logging.getLogger(__name__)

    if USE_PI_CAMERA:
        try:
            from picamera2 import Picamera2

            logger.info("Initializing Pi Camera...")
            camera = Picamera2()
            camera_config = camera.create_still_configuration(
                main={"size": CAMERA_RESOLUTION}
            )
            camera.configure(camera_config)
            camera.start()
            time.sleep(2)  # Warm up camera
            logger.info(f"Pi Camera ready ({CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]})")
            return camera, "picamera"

        except ImportError:
            logger.warning("picamera2 not available. Falling back to USB webcam.")
        except Exception as e:
            logger.warning(f"Pi Camera failed: {e}. Falling back to USB webcam.")

    # USB webcam fallback
    import cv2
    logger.info(f"Opening USB webcam (index {USB_CAMERA_INDEX})...")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Could not open webcam!")
        return None, None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    logger.info("USB webcam ready")
    return cap, "usb"


def capture_frame(camera, camera_type):
    """Capture a frame from the camera."""
    if camera_type == "picamera":
        import numpy as np
        frame = camera.capture_array()
        return frame
    elif camera_type == "usb":
        ret, frame = camera.read()
        return frame if ret else None
    return None


def run_detection_loop(args):
    """Main detection loop."""
    logger = logging.getLogger(__name__)

    print()
    print("🚁" + "=" * 56 + "🚁")
    print("   Disaster Rescue Drone — Person Detection System")
    print("   Final Year Project")
    print("🚁" + "=" * 56 + "🚁")
    print()

    # ── Step 1: Initialize ML Model ──
    logger.info("Step 1: Loading person detection model...")
    detector = PersonDetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
        target_class=TARGET_CLASS,
    )
    print("  ✅ Model loaded\n")

    # ── Step 2: Initialize Camera ──
    logger.info("Step 2: Setting up camera...")
    camera, camera_type = setup_camera()
    if camera is None:
        if args.simulate:
            logger.info("Simulate mode — no camera needed.")
            camera_type = "simulate"
        else:
            logger.error("No camera available. Exiting.")
            return
    print(f"  ✅ Camera ready ({camera_type})\n")

    # ── Step 3: Connect to Drone ──
    drone = None
    if not args.test_camera and not args.simulate:
        logger.info("Step 3: Connecting to Pixhawk 2.4.8...")
        drone = DroneController(
            connection_string=PIXHAWK_CONNECTION,
            baud_rate=PIXHAWK_BAUD,
            timeout=PIXHAWK_TIMEOUT,
        )
        if drone.connect():
            print("  ✅ Pixhawk connected\n")
        else:
            logger.error("Failed to connect to Pixhawk. Running in camera-only mode.")
            drone = None
    else:
        logger.info("Step 3: Skipping drone connection (test/simulate mode)")
        print("  ⏭️  Drone connection skipped\n")

    # ── Step 4: Initialize GPS Reporter ──
    reporter = GPSReporter(
        log_file=GPS_LOG_FILE,
        enable_csv=ENABLE_CSV_LOGGING,
        enable_console=ENABLE_CONSOLE_ALERTS,
        save_images=SAVE_DETECTION_IMAGES,
        images_dir=DETECTION_IMAGES_DIR,
    )
    print("  ✅ GPS reporter initialized\n")

    # ── Step 5: Detection Loop ──
    print("=" * 60)
    print("  🔍 Starting detection loop... (Ctrl+C to stop)")
    print("=" * 60)
    print()

    consecutive_detections = 0
    frame_count = 0
    detection_active = False

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            if camera_type == "simulate":
                import numpy as np
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            else:
                frame = capture_frame(camera, camera_type)

            if frame is None:
                logger.warning("Failed to capture frame. Retrying...")
                time.sleep(0.5)
                continue

            frame_count += 1

            # Run detection
            if DISPLAY_PREVIEW:
                detections, annotated = detector.detect_with_annotated_frame(frame)
            else:
                detections = detector.detect(frame)
                annotated = None

            persons = [d for d in detections]
            num_persons = len(persons)

            # Status update (every 10 frames)
            if frame_count % 10 == 0:
                mode = drone.get_mode() if drone else "N/A"
                logger.info(f"Frame #{frame_count} | Persons: {num_persons} | "
                            f"Mode: {mode} | Consecutive: {consecutive_detections}")

            # Detection logic
            if num_persons > 0:
                consecutive_detections += 1
                logger.info(f"👤 Person(s) detected! Count: {num_persons}, "
                            f"Consecutive: {consecutive_detections}/{MIN_DETECTION_COUNT}")
            else:
                consecutive_detections = 0

            # Trigger hover when enough consecutive detections
            if consecutive_detections >= MIN_DETECTION_COUNT and not detection_active:
                detection_active = True
                logger.info(f"🚨 CONFIRMED DETECTION after {MIN_DETECTION_COUNT} frames!")

                # Get GPS
                gps = None
                drone_info = None
                if drone:
                    gps = drone.get_gps()
                    drone_info = {
                        "heading": drone.get_heading(),
                        "groundspeed": drone.get_groundspeed(),
                        "battery_level": drone.get_battery().get("level") if drone.get_battery() else None,
                    }

                # Report detection
                reporter.report_detection(
                    gps_data=gps,
                    detections=persons,
                    drone_info=drone_info,
                    frame=annotated if annotated is not None else frame,
                )

                # Hover
                if drone:
                    drone.hover()
                    logger.info(f"⏸️ Hovering for {HOVER_DURATION} seconds...")
                    time.sleep(HOVER_DURATION)

                    # Resume mission
                    logger.info(f"Cooldown: {COOLDOWN_AFTER_DETECTION}s...")
                    time.sleep(COOLDOWN_AFTER_DETECTION)
                    drone.resume_mission()
                else:
                    logger.info(f"[SIM] Would hover for {HOVER_DURATION}s (no drone connected)")
                    time.sleep(2)  # Brief pause in test mode

                detection_active = False
                consecutive_detections = 0

            # Display preview if enabled
            if DISPLAY_PREVIEW and annotated is not None:
                import cv2
                cv2.imshow("Detection Preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Control detection rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, DETECTION_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n🛑 Detection loop stopped by user.")

    finally:
        # Cleanup
        logger.info("Cleaning up...")

        if camera_type == "usb" and camera is not None:
            camera.release()
        elif camera_type == "picamera" and camera is not None:
            camera.stop()

        if drone:
            drone.disconnect()

        if DISPLAY_PREVIEW:
            import cv2
            cv2.destroyAllWindows()

        # Print summary
        summary = reporter.get_summary()
        print()
        print("=" * 60)
        print("  📊 Session Summary")
        print("=" * 60)
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Detection log: {summary['log_file']}")
        print("=" * 60)


def main():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Disaster Rescue Drone — Person Detection System"
    )
    parser.add_argument(
        "--test-camera", action="store_true",
        help="Camera test only — no drone connection"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Simulate mode — no hardware required"
    )
    args = parser.parse_args()

    setup_logging()
    run_detection_loop(args)


if __name__ == "__main__":
    main()
