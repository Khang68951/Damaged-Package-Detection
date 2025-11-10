from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
from datetime import datetime
import random
import os

app = Flask(__name__)
CORS(app)

detector = None
CURRENT_THRESHOLD = 0.3

def initialize_model():
    global detector
    try:
        from model_loader import DamageDetector
        
        MODEL_PATH = "best_model.pth"
        
        if os.path.exists(MODEL_PATH):
            detector = DamageDetector(MODEL_PATH)
        else:
            print(f"Model file not found: {MODEL_PATH}")
            detector = None
            
    except Exception as e:
        print(f"Error initializing model: {e}")
        detector = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None,
        "current_threshold": CURRENT_THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict_damage():
    global CURRENT_THRESHOLD
    
    try:
        if detector is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file format"}), 400
        
        pkg_id = request.form.get('pkg_id', f"pkg_{random.randint(1000, 9999)}")
        threshold = float(request.form.get('threshold', CURRENT_THRESHOLD))
        
        if 'threshold' in request.form:
            CURRENT_THRESHOLD = threshold
        
        image_bytes = file.read()
        detection_result = detector.predict(image_bytes, CURRENT_THRESHOLD)
        
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        response_data = {
            "created_at": created_at,
            "pkg_id": pkg_id,
            "damaged": detection_result["damaged"],
            "damaged_confidence": detection_result["damaged_confidence"],
            "valid": detection_result["valid"],
            "valid_confidence": detection_result["valid_confidence"],
            "threshold_used": CURRENT_THRESHOLD,
            "damage_boxes": detection_result["damage_boxes"],
            "package_boxes": detection_result["package_boxes"]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/threshold', methods=['POST', 'GET'])
def manage_threshold():
    global CURRENT_THRESHOLD
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_threshold = float(data.get('threshold', CURRENT_THRESHOLD))
            
            if 0 <= new_threshold <= 1:
                CURRENT_THRESHOLD = new_threshold
                return jsonify({
                    "status": "success",
                    "new_threshold": CURRENT_THRESHOLD
                })
            else:
                return jsonify({"error": "Threshold must be between 0 and 1"}), 400
                
        except Exception as e:
            return jsonify({"error": f"Invalid threshold: {str(e)}"}), 400
    
    else:
        return jsonify({"current_threshold": CURRENT_THRESHOLD})

@app.route('/camera')
def camera_detection():
    return render_template('camera_detection.html')

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "message": "API is working",
        "model_loaded": detector is not None,
        "current_threshold": CURRENT_THRESHOLD
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Package Damage Detection API",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict damage from image",
            "GET/POST /threshold": "Manage confidence threshold",
            "GET /camera": "Camera detection interface"
        }
    })

if __name__ == '__main__':
    print("Starting Package Damage Detection API...")
    initialize_model()
    app.run(host='0.0.0.0', port=5000, debug=True)