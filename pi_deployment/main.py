"""
Main Entry Point — Drone Person Detection System
==================================================
This is the main script that runs on the Raspberry Pi 5.
It combines the detector, drone controller, video stream,
and dual-channel communications into a single detection loop.

Architecture:
    - MAIN THREAD: Captures frames at full speed, streams to browser
    - DETECTION THREAD: Runs YOLO inference in background, no lag

Flow:
    1. Connect to Pixhawk 2.4.8
    2. Initialize camera, ML model, and video stream
    3. Stream smooth live feed to browser
    4. Detect persons in background -> hover -> report GPS
    5. Resume mission after hover duration

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
import threading
import cv2
from datetime import datetime

# Import modules
from config import (
    PIXHAWK_CONNECTION, PIXHAWK_BAUD, PIXHAWK_TIMEOUT,
    MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_CLASS,
    DETECTION_INTERVAL, HOVER_DURATION, COOLDOWN_AFTER_DETECTION,
    MIN_DETECTION_COUNT, SAVE_DETECTION_IMAGES, DETECTION_IMAGES_DIR,
    CAMERA_RESOLUTION, USE_PI_CAMERA, USB_CAMERA_INDEX,
    LOG_LEVEL, STREAM_HOST, STREAM_PORT, ENABLE_VIDEO_STREAM,
    ENABLE_MAVLINK_ALERTS, ENABLE_HTTP_ALERTS, GROUND_STATION_URL,
)
from detector import PersonDetector
from drone_controller import DroneController
from communications import Communications
from video_stream import VideoStream


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
        frame = camera.capture_array()
        return frame
    elif camera_type == "usb":
        ret, frame = camera.read()
        return frame if ret else None
    return None


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on a frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        # Draw green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        label = f"Person {conf:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return annotated


def run_detection_loop(args):
    """Main detection loop with threaded YOLO for smooth video streaming."""
    logger = logging.getLogger(__name__)

    print()
    print("=" * 60)
    print("   Disaster Rescue Drone — Person Detection System")
    print("   Final Year Project")
    print("=" * 60)
    print()

    # ── Step 1: Initialize ML Model ──
    logger.info("Step 1: Loading person detection model...")
    detector = PersonDetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
        target_class=TARGET_CLASS,
    )
    print("  Model loaded\n")

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
    print(f"  Camera ready ({camera_type})\n")

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
            print("  Pixhawk connected\n")
        else:
            logger.error("Failed to connect to Pixhawk. Running in camera-only mode.")
            drone = None
    else:
        logger.info("Step 3: Skipping drone connection (test/simulate mode)")
        print("  Drone connection skipped\n")

    # ── Step 4: Initialize Dual-Channel Communications ──
    logger.info("Step 4: Setting up communications...")
    comms = Communications(
        ground_station_url=GROUND_STATION_URL,
        enable_mavlink=ENABLE_MAVLINK_ALERTS,
        enable_http=ENABLE_HTTP_ALERTS,
        save_local=SAVE_DETECTION_IMAGES,
        detections_dir=DETECTION_IMAGES_DIR,
    )
    print("  Communications initialized\n")

    # ── Step 5: Start Video Stream Server ──
    stream = None
    if ENABLE_VIDEO_STREAM:
        logger.info("Step 5: Starting live video stream...")
        stream = VideoStream(host=STREAM_HOST, port=STREAM_PORT)
        if stream.start():
            print(f"  Live feed at http://<Pi_IP>:{STREAM_PORT}\n")
        else:
            logger.warning("Video stream failed to start. Continuing without it.")
            stream = None
    else:
        print("  Video stream disabled\n")

    # ── Step 6: Detection Loop (Threaded Architecture) ──
    print("=" * 60)
    print("  Starting detection loop... (Ctrl+C to stop)")
    print("=" * 60)
    print()

    # Shared state between main thread and detection thread
    latest_frame = None
    latest_detections = []
    latest_annotated = None
    frame_lock = threading.Lock()
    detection_lock = threading.Lock()
    running = True
    frame_count = 0
    consecutive_detections = 0
    detection_active = False

    def detection_worker():
        """Background thread: runs YOLO on frames at its own pace."""
        nonlocal latest_detections, latest_annotated, consecutive_detections
        nonlocal detection_active, running

        while running:
            # Grab the latest frame
            with frame_lock:
                frame_to_process = latest_frame

            if frame_to_process is None:
                time.sleep(0.05)
                continue

            # Run YOLO detection (this is the slow part — ~200-500ms)
            detections = detector.detect(frame_to_process)
            persons = [d for d in detections]
            num_persons = len(persons)

            # Draw bounding boxes on a copy of the frame
            if num_persons > 0:
                annotated = draw_detections(frame_to_process, persons)
            else:
                annotated = frame_to_process

            # Update shared detection results
            with detection_lock:
                latest_detections = persons
                latest_annotated = annotated

            # Detection logic
            if num_persons > 0:
                consecutive_detections += 1
                logger.info(f"Person(s) detected! Count: {num_persons}, "
                            f"Consecutive: {consecutive_detections}/{MIN_DETECTION_COUNT}")
            else:
                consecutive_detections = 0

            # Trigger hover when enough consecutive detections
            if consecutive_detections >= MIN_DETECTION_COUNT and not detection_active:
                detection_active = True
                logger.info(f"CONFIRMED DETECTION after {MIN_DETECTION_COUNT} frames!")

                # Get drone info
                gps = None
                drone_info = None
                if drone:
                    gps = drone.get_gps()
                    drone_info = {
                        "heading": drone.get_heading(),
                        "groundspeed": drone.get_groundspeed(),
                        "battery_level": drone.get_battery().get("level") if drone.get_battery() else None,
                    }

                # Send alerts through all channels
                comms.send_alert(
                    gps_data=gps,
                    detections=persons,
                    drone_controller=drone,
                    frame=annotated,
                    drone_info=drone_info,
                )

                # Command drone to hover
                if drone:
                    drone.hover()
                    logger.info(f"Hovering for {HOVER_DURATION} seconds...")
                    time.sleep(HOVER_DURATION)

                    # Resume mission
                    logger.info(f"Cooldown: {COOLDOWN_AFTER_DETECTION}s...")
                    time.sleep(COOLDOWN_AFTER_DETECTION)
                    drone.resume_mission()
                else:
                    logger.info(f"[SIM] Would hover for {HOVER_DURATION}s (no drone)")
                    time.sleep(2)  # Brief pause in test mode

                detection_active = False
                consecutive_detections = 0

            # Control detection rate
            time.sleep(DETECTION_INTERVAL)

    # Start the detection thread
    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()
    logger.info("Detection thread started (YOLO runs in background)")

    try:
        while True:
            # ── Main Thread: Capture + Stream at full speed ──
            if camera_type == "simulate":
                import numpy as np
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            else:
                frame = capture_frame(camera, camera_type)

            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # Give the latest frame to the detection thread
            with frame_lock:
                latest_frame = frame

            # Get the latest detection results (non-blocking)
            with detection_lock:
                current_detections = latest_detections
                display_frame = latest_annotated if latest_annotated is not None else frame

            # Get GPS data
            gps = None
            if drone:
                gps = drone.get_gps()

            # Update the live video stream (smooth, no YOLO blocking)
            if stream:
                stream.update_frame(display_frame, detections=current_detections, gps_data=gps)

            # Status update (every 30 frames)
            if frame_count % 30 == 0:
                mode = drone.get_mode() if drone else "N/A"
                logger.info(f"Frame #{frame_count} | Persons: {len(current_detections)} | "
                            f"Mode: {mode}")

            # ~15 FPS streaming rate (smooth video)
            time.sleep(0.066)

    except KeyboardInterrupt:
        print("\n\nDetection loop stopped by user.")

    finally:
        # Cleanup
        running = False
        logger.info("Cleaning up...")

        if stream:
            stream.stop()

        if camera_type == "usb" and camera is not None:
            camera.release()
        elif camera_type == "picamera" and camera is not None:
            camera.stop()

        if drone:
            drone.disconnect()

        # Print summary
        summary = comms.get_summary()
        print()
        print("=" * 60)
        print("  Session Summary")
        print("=" * 60)
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Detection images: {summary['detections_dir']}")
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
