from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
class HumanDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 model.
        Using 'yolov8n.pt' (nano) for speed, or 'yolov8s.pt' (small) for better accuracy.
        """
        self.model = YOLO(model_path)
    def detect(self, image):
        """
        Detects humans in the given image.
        Args:
            image: PIL Image or numpy array.
        Returns:
            processed_image: Image with bounding boxes drawn.
            human_found: Boolean indicating if a human was detected.
            detections: List of detection details usually needed for further processing (optional).
        """
        # Run inference
        results = self.model(image, classes=[0], conf=0.4) # class 0 is person in COCO
        # We only care about the first result for a single image
        result = results[0]
        
        # Plot predictions on the image
        # This returns a BGR numpy array
        processed_image_bgr = result.plot()
        
        # Convert BGR to RGB for Streamlit/PIL
        processed_image_rgb = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
        
        human_found = len(result.boxes) > 0
        
        return processed_image_rgb, human_found