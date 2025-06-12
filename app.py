from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Path to the saved model (adjust path if needed)
model_path = "./model/ai_vs_real_image_detection/model3.0/model_weights.pth"

try:
    # Load the RegNet model architecture
    model = models.regnet_y_32gf(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    # Define the same transforms used for validation/testing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logging.info("Model loaded successfully!")
    
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request"}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")
        
        # Process the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move to same device as model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Map class ID to label
        labels = ["Real Image", "AI Generated Image"]
        predicted_label = labels[predicted_class_id]
        
        logging.info(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}")
        
        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence
        })
            
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)