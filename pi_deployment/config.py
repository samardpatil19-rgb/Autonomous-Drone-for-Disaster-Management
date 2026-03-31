"""
Pi Deployment Configuration
============================
Settings for the Raspberry Pi 5 deployment.
Modify these values to match your hardware setup.
"""

# ============================================================
# PIXHAWK CONNECTION (Pixhawk 2.4.8)
# ============================================================
# UART connection from Pi 5 GPIO to Pixhawk TELEM2
PIXHAWK_CONNECTION = "/dev/serial0"      # Pi UART
PIXHAWK_BAUD = 57600                      # Default MAVLink baud rate
PIXHAWK_TIMEOUT = 30                      # Connection timeout (seconds)

# ============================================================
# CAMERA
# ============================================================
CAMERA_RESOLUTION = (640, 480)            # Width x Height
CAMERA_FORMAT = "RGB888"                  # Camera pixel format
USE_PI_CAMERA = True                      # True = Pi Camera, False = USB webcam
USB_CAMERA_INDEX = 0                      # USB webcam index (if USE_PI_CAMERA=False)

# ============================================================
# DETECTION
# ============================================================
MODEL_PATH = "model/best_ncnn_model"      # Path to exported NCNN model on Pi
CONFIDENCE_THRESHOLD = 0.5                # Minimum confidence for person detection
TARGET_CLASS = 0                          # Class ID for person
DETECTION_INTERVAL = 0.5                  # Seconds between detection runs (2 FPS)

# ============================================================
# DRONE BEHAVIOR
# ============================================================
HOVER_DURATION = 15                       # Seconds to hover after detection
COOLDOWN_AFTER_DETECTION = 5              # Seconds before resuming mission after hover
MIN_DETECTION_COUNT = 2                   # Minimum consecutive detections to trigger hover
                                          # (reduces false positives)

# ============================================================
# VIDEO STREAMING (Flask server on the Pi)
# ============================================================
STREAM_HOST = "0.0.0.0"                   # Bind to all interfaces
STREAM_PORT = 5000                        # Access at http://<Pi_IP>:5000
ENABLE_VIDEO_STREAM = True                # Enable live video feed

# ============================================================
# COMMUNICATIONS (Dual-channel alerts)
# ============================================================
ENABLE_MAVLINK_ALERTS = True              # Channel 1: MAVLink via telemetry radio
ENABLE_HTTP_ALERTS = False                # Channel 2: HTTP via 4G/ZeroTier
GROUND_STATION_URL = None                 # Set to http://<laptop_zerotier_ip>:8080/alert
SAVE_DETECTION_IMAGES = True              # Save annotated frames to SD card
DETECTION_IMAGES_DIR = "detections/"      # Directory for saved detection images

# ============================================================
# SYSTEM
# ============================================================
LOG_LEVEL = "INFO"                        # Logging level (DEBUG, INFO, WARNING, ERROR)
