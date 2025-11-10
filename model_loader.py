import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io

class DamageDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'roi_heads.box_predictor.cls_score.weight' in checkpoint:
                num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
            else:
                num_classes = 3
            
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                min_size=800,
                max_size=1333
            )
            
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.model.to(self.device)
            
        except Exception as e:
            print(f"Model loading error: {e}")
            raise e

    def preprocess_image(self, image):
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        
        original_size = image.size
        processed_image = self.transform(image)
        return processed_image, original_size

    def predict(self, image, threshold=0.3):
        try:
            if self.model is None:
                return self._empty_result()
            
            input_tensor, original_size = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                predictions = self.model([input_tensor])
            
            return self.process_predictions(predictions, original_size, threshold)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._empty_result()

    def _empty_result(self):
        return {
            "damaged": False,
            "damaged_confidence": 0.0,
            "valid": True,
            "valid_confidence": 1.0,
            "damage_boxes": [],
            "package_boxes": []
        }

    def process_predictions(self, predictions, original_size, threshold):
        original_width, original_height = original_size
        
        if not predictions or len(predictions) == 0:
            return self._empty_result()
        
        pred = predictions[0]
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        if len(boxes) == 0:
            return self._empty_result()
        
        damage_boxes = []
        package_boxes = []
        damage_confidences = []
        
        for box, score, label in zip(boxes, scores, labels):
            if score.item() > threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                
                scale_x = original_width / 800.0
                scale_y = original_height / 800.0
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                box_data = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(score.item(), 4),
                    "label": label.item()
                }
                
                if label.item() == 1:
                    box_data["label_name"] = "damage"
                    damage_boxes.append(box_data)
                    damage_confidences.append(score.item())
                elif label.item() == 2:
                    box_data["label_name"] = "package"
                    package_boxes.append(box_data)
        
        damage_boxes.sort(key=lambda x: x["confidence"], reverse=True)
        package_boxes.sort(key=lambda x: x["confidence"], reverse=True)
        
        has_damage = len(damage_boxes) > 0
        max_damage_conf = max(damage_confidences) if damage_confidences else 0.0
        
        return {
            "damaged": has_damage,
            "damaged_confidence": round(max_damage_conf, 4),
            "valid": True,
            "valid_confidence": 1.0,
            "damage_boxes": damage_boxes,
            "package_boxes": package_boxes
        }