from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import base64
import io
import pyttsx3
import time
from threading import Thread

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Model configuration
MODEL_PATH = os.path.join('app', 'model', 'model.pt')
IMG_SIZE = 224

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def pspi_to_class(pspi):
    if pspi < 0.5:
        return "No pain"
    elif pspi < 1.5:
        return "Very mild"
    elif pspi < 2.5:
        return "Mild"
    elif pspi < 3.5:
        return "Moderate"
    elif pspi < 4.5:
        return "Moderately severe"
    elif pspi < 5.5:
        return "Severe"
    else:
        return "Very severe"

def speak_pain_level(pain_class, pain_level):
    # create and configure a fresh TTS engine each time
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    text = f"This person has {pain_class} level of pain, with a pain score of {pain_level} percent"
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak_after_delay(pain_class, pain_level):
    time.sleep(1)
    speak_pain_level(pain_class, pain_level)

# Load model at startup
model = load_model()

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            pain_score = output.item()
            pain_class = pspi_to_class(pain_score)
        
        confidence = 85
        pain_level_percentage = round(pain_score * 100 / 6)
        
        Thread(
            target=speak_after_delay,
            args=(pain_class, pain_level_percentage),
            daemon=True
        ).start()
        
        return jsonify({
            'pain_level': pain_level_percentage,
            'pain_class': pain_class,
            'confidence': confidence
        })
        
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    except OSError:
        app.logger.info("Port 5001 in use, trying 5002...")
        try:
            app.run(debug=True, port=5002)
        except OSError:
            app.logger.info("Ports 5001 & 5002 in use, using random port")
            app.run(debug=True, port=0)
