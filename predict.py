from cog import BasePredictor, Input, Path as CogPath
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import cv2

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.model = YOLO("my_model.pt")

    def predict(
        self,
        image: CogPath = Input(description="Input face image"),
        confidence_threshold: float = Input(
            description="Confidence threshold for detections",
            default=0.25,
            ge=0.0,
            le=1.0
        ),
        return_json: bool = Input(
            description="Return JSON data instead of annotated image",
            default=False
        )
    ):
        """Run face skin detection"""
        
        # Load image
        img = Image.open(image)
        
        # Run inference
        results = self.model(img, conf=confidence_threshold)
        
        # If user wants JSON output
        if return_json:
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = results[0].names[class_id]
                bbox = box.xyxy[0].tolist()
                
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
        
        # Otherwise return annotated image
        else:
            # Get annotated image from YOLO (with boxes drawn)
            annotated_img = results[0].plot()
            
            # Convert BGR to RGB (YOLO uses BGR)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            output_img = Image.fromarray(annotated_img)
            
            # Save the annotated image
            output_path = Path("/tmp/output.jpg")
            output_img.save(output_path, format="JPEG", quality=95)
            
            # Return the annotated image
            return CogPath(output_path)
