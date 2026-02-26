"""
Person Detector Module
=======================
Handles YOLOv8 model loading and inference for person detection.
Optimized for Raspberry Pi 5 with NCNN backend.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLOv8n-based person detector for drone aerial imagery."""

    def __init__(self, model_path, confidence=0.5, target_class=0):
        """
        Initialize the detector.

        Args:
            model_path: Path to the exported model (NCNN, ONNX, or PyTorch)
            confidence: Minimum confidence threshold for detections
            target_class: Class ID to detect (0 = person)
        """
        self.confidence = confidence
        self.target_class = target_class
        self.model = None

        logger.info(f"Loading model from: {model_path}")
        start = time.time()

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            load_time = time.time() - start
            logger.info(f"Model loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame):
        """
        Run person detection on a single frame.

        Args:
            frame: numpy array (BGR or RGB image)

        Returns:
            list of dict: Detected persons with keys:
                - bbox: (x1, y1, x2, y2) bounding box
                - confidence: detection confidence score
                - center: (cx, cy) center point of bounding box
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return []

        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())

            # Only return person detections
            if cls_id == self.target_class:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(conf, 3),
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                })

        return detections

    def detect_with_annotated_frame(self, frame):
        """
        Run detection and return both detections and annotated frame.

        Args:
            frame: numpy array (BGR or RGB image)

        Returns:
            tuple: (detections_list, annotated_frame)
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            if cls_id == self.target_class:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(conf, 3),
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                })

        annotated = results[0].plot()
        return detections, annotated
