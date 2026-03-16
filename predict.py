from cog import BasePredictor, Input, Path
from ultralytics import YOLO
from PIL import Image
import json

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.model = YOLO("my_model.pt")

    def predict(
        self,
        image: Path = Input(description="Input face image"),
        confidence_threshold: float = Input(
            description="Confidence threshold for detections",
            default=0.25,
            ge=0.0,
            le=1.0
        )
    ) -> dict:
        """Run face skin detection"""
        
        # Load image
        img = Image.open(image)
        
        # Run inference
        results = self.model(img, conf=confidence_threshold)
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = results[0].names[class_id]
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bounding_box": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                }
            })
        
        return {
            "detections": detections,
            "count": len(detections)
        }
