"""
Script 06: Test Detection with Laptop Webcam
==============================================
Opens the laptop webcam and runs real-time person detection
using the trained YOLOv8n model. Great for quick visual validation.

Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot

Usage:
    python scripts/06_test_webcam.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import (
    get_best_model_path, CONFIDENCE_THRESHOLD,
    WEBCAM_INDEX, WEBCAM_DISPLAY_WIDTH, WEBCAM_DISPLAY_HEIGHT,
    PROJECT_ROOT
)


def test_webcam():
    """Test person detection using laptop webcam."""
    import cv2
    from ultralytics import YOLO

    print("=" * 60)
    print("  Webcam Detection Test — Person Detector")
    print("=" * 60)
    print()

    # Load model
    model_path = get_best_model_path()
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python scripts/03_train_model.py' first.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded!\n")

    # Open webcam
    print(f"Opening webcam (index {WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Try changing WEBCAM_INDEX in config/training_config.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_DISPLAY_HEIGHT)

    print("Webcam opened!")
    print()
    print("Controls:")
    print("  'q' — Quit")
    print("  's' — Save screenshot")
    print()

    # Create screenshots directory
    screenshots_dir = os.path.join(PROJECT_ROOT, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    frame_count = 0
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Run detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        frame_count += 1

        # Calculate FPS
        elapsed = time.time() - fps_start
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Count detections
        person_count = sum(1 for box in results[0].boxes if int(box.cls) == 0)

        # Add info overlay
        info_text = f"FPS: {fps:.1f} | Persons: {person_count} | Conf: {CONFIDENCE_THRESHOLD}"
        cv2.rectangle(annotated_frame, (0, 0), (500, 35), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame, info_text,
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        # Alert if person detected
        if person_count > 0:
            alert = f"PERSON DETECTED! ({person_count})"
            cv2.rectangle(annotated_frame, (0, 40), (400, 80), (0, 0, 255), -1)
            cv2.putText(
                annotated_frame, alert,
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

        # Display
        cv2.imshow("Drone Person Detector — Press 'q' to quit", annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshots_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam test ended.")


if __name__ == "__main__":
    test_webcam()
