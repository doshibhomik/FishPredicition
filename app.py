from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the model, scaler, and label encoder when the app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def classify_fish(length1, length2, length3, height, width):
    # Prepare the input data
    input_data = np.array([[length1, length2, length3, height, width]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    predicted_species = model.predict(input_data_scaled)
    
    # Decode the predicted species
    predicted_species_name = label_encoder.inverse_transform(predicted_species)[0]
    
    return predicted_species_name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    length1 = data['length1']
    length2 = data['length2']
    length3 = data['length3']
    height = data['height']
    width = data['width']
    
    # Make the prediction
    species = classify_fish(length1, length2, length3, height, width)
    image_url = url_for('static', filename=f'{species.lower()}.png')
    
    return jsonify({'species': species, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
