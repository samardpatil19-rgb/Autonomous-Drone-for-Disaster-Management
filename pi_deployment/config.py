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
# GPS REPORTING
# ============================================================
GPS_LOG_FILE = "detection_log.csv"        # CSV file for logging detections
ENABLE_CONSOLE_ALERTS = True              # Print alerts to console
ENABLE_CSV_LOGGING = True                 # Log detections to CSV

# ============================================================
# SYSTEM
# ============================================================
LOG_LEVEL = "INFO"                        # Logging level (DEBUG, INFO, WARNING, ERROR)
DISPLAY_PREVIEW = False                   # Show camera preview (only if display connected)
SAVE_DETECTION_IMAGES = True              # Save frames with detections
DETECTION_IMAGES_DIR = "detections/"      # Directory for saved detection images
