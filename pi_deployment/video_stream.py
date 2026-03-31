"""
Video Stream Server
====================
Lightweight Flask server that streams the live detection feed
(with YOLO bounding boxes) to any browser over the network.

Access the stream by opening a browser and navigating to:
    http://<Pi_ZeroTier_IP>:5000

Works over home Wi-Fi, college Wi-Fi, or 4G LTE dongle
when both devices are on the same ZeroTier network.
"""

import threading
import time
import logging
import cv2

logger = logging.getLogger(__name__)


class VideoStream:
    """MJPEG video streaming server using Flask."""

    def __init__(self, host="0.0.0.0", port=5000):
        """
        Initialize the video stream server.

        Args:
            host: IP to bind to (0.0.0.0 = all interfaces)
            port: Port to serve on (default 5000)
        """
        self.host = host
        self.port = port
        self.current_frame = None
        self.lock = threading.Lock()
        self.app = None
        self.server_thread = None
        self.running = False
        self.latest_detections = []
        self.latest_gps = None

    def start(self):
        """Start the Flask streaming server in a background thread."""
        try:
            from flask import Flask, Response, render_template_string

            self.app = Flask(__name__)

            # Suppress Flask's default request logging
            flask_log = logging.getLogger('werkzeug')
            flask_log.setLevel(logging.WARNING)

            stream_ref = self  # Reference for closures

            @self.app.route('/')
            def index():
                """Main page with embedded video stream and detection info."""
                return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Drone Live Feed</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {
                            background: #0a0a0a; color: #e0e0e0;
                            font-family: 'Segoe UI', Arial, sans-serif;
                            margin: 0; padding: 20px;
                            display: flex; flex-direction: column;
                            align-items: center;
                        }
                        h1 {
                            color: #ff4444; font-size: 1.5em;
                            border-bottom: 2px solid #ff4444;
                            padding-bottom: 10px; margin-bottom: 20px;
                        }
                        .status {
                            background: #1a1a2e; border: 1px solid #333;
                            border-radius: 8px; padding: 12px 24px;
                            margin-bottom: 15px; font-size: 0.9em;
                        }
                        .status span { color: #00ff88; font-weight: bold; }
                        img {
                            max-width: 100%; border: 2px solid #333;
                            border-radius: 8px;
                        }
                        .alert {
                            background: #ff4444; color: white;
                            padding: 10px 20px; border-radius: 8px;
                            margin-top: 10px; font-weight: bold;
                            display: none;
                        }
                    </style>
                    <script>
                        setInterval(function() {
                            fetch('/status')
                                .then(r => r.json())
                                .then(data => {
                                    document.getElementById('det-count').textContent = data.detections;
                                    document.getElementById('gps-info').textContent = data.gps;
                                    let alert = document.getElementById('alert-box');
                                    if (data.detections > 0) {
                                        alert.style.display = 'block';
                                        alert.textContent = 'PERSON DETECTED! ' + data.gps;
                                    } else {
                                        alert.style.display = 'none';
                                    }
                                });
                        }, 1000);
                    </script>
                </head>
                <body>
                    <h1>DISASTER RESCUE DRONE — LIVE FEED</h1>
                    <div class="status">
                        Persons in Frame: <span id="det-count">0</span>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        GPS: <span id="gps-info">Waiting...</span>
                    </div>
                    <div id="alert-box" class="alert"></div>
                    <img src="/video_feed" alt="Live Feed">
                </body>
                </html>
                ''')

            @self.app.route('/video_feed')
            def video_feed():
                """MJPEG video stream endpoint."""
                return Response(
                    stream_ref._generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )

            @self.app.route('/status')
            def status():
                """JSON status endpoint for AJAX updates."""
                import json
                gps_str = "N/A"
                if stream_ref.latest_gps:
                    gps_str = (f"Lat: {stream_ref.latest_gps['lat']:.6f}, "
                               f"Lon: {stream_ref.latest_gps['lon']:.6f}")
                return json.dumps({
                    "detections": len(stream_ref.latest_detections),
                    "gps": gps_str,
                })

            # Start Flask in a background thread
            self.running = True
            self.server_thread = threading.Thread(
                target=self.app.run,
                kwargs={"host": self.host, "port": self.port, "threaded": True},
                daemon=True,
            )
            self.server_thread.start()
            logger.info(f"Video stream server started at http://{self.host}:{self.port}")
            return True

        except ImportError:
            logger.error("Flask not installed! Run: pip install flask")
            return False
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return False

    def update_frame(self, frame, detections=None, gps_data=None):
        """
        Update the current frame being streamed.

        Args:
            frame: numpy array (BGR image with bounding boxes already drawn)
            detections: list of detection dicts
            gps_data: dict with 'lat', 'lon', 'alt'
        """
        with self.lock:
            self.current_frame = frame.copy()
            self.latest_detections = detections or []
            self.latest_gps = gps_data

    def _generate_frames(self):
        """Generator that yields MJPEG frames for the video stream."""
        while self.running:
            with self.lock:
                if self.current_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.current_frame.copy()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

            time.sleep(0.033)  # ~30 FPS max

    def stop(self):
        """Stop the streaming server."""
        self.running = False
        logger.info("Video stream server stopped.")
