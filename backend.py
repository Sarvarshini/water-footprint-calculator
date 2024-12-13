from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS  # To handle CORS issues when connecting frontend

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = load_model('my_model.h5')

# Define class labels and water footprint data
class_names = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 
               'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
               'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

water_footprint_data = {
    "Bean": "Beans have a water footprint of 1500 liters/kg. An alternative is Lentils with 600 liters/kg.",
    "Bitter Gourd": "Bitter gourd has a water footprint of 2000 liters/kg. An alternative is Cucumber with 500 liters/kg.",
    "Bottle Gourd": "Bottle gourd has a water footprint of 1800 liters/kg. An alternative is Zucchini with 700 liters/kg.",
    "Brinjal": "Brinjal has a water footprint of 1800 liters/kg. An alternative is Bell Pepper with 500 liters/kg.",
    "Broccoli": "Broccoli has a water footprint of 1000 liters/kg. An alternative is Kale with 600 liters/kg.",
    "Cabbage": "Cabbage has a water footprint of 700 liters/kg. An alternative is Swiss Chard with 500 liters/kg.",
    "Capsicum": "Capsicum has a water footprint of 500 liters/kg. An alternative is Chili Peppers with 400 liters/kg.",
    "Carrot": "Carrot has a water footprint of 600 liters/kg. An alternative is Beetroot with 500 liters/kg.",
    "Cauliflower": "Cauliflower has a water footprint of 1200 liters/kg. An alternative is Romanesco with 1000 liters/kg.",
    "Cucumber": "Cucumber has a water footprint of 500 liters/kg. An alternative is Celery with 300 liters/kg.",
    "Papaya": "Papaya has a water footprint of 2500 liters/kg. An alternative is Melon with 1000 liters/kg.",
    "Potato": "Potato has a water footprint of 500 liters/kg. An alternative is Sweet Potato with 400 liters/kg.",
    "Pumpkin": "Pumpkin has a water footprint of 1200 liters/kg. An alternative is Butternut Squash with 1000 liters/kg.",
    "Radish": "Radish has a water footprint of 300 liters/kg. An alternative is Turnip with 250 liters/kg.",
    "Tomato": "Tomato has a water footprint of 1000 liters/kg. An alternative is Cherry Tomato with 800 liters/kg."
}

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match model's input shape
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# API endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        img = Image.open(file)
        img_preprocessed = preprocess_image(img)
        prediction = model.predict(img_preprocessed)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        footprint_info = water_footprint_data.get(predicted_class, "No data available")
        return jsonify({
            "predicted_class": predicted_class,
            "footprint_info": footprint_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
