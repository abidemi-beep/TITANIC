from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'titanic_survival_model.pkl')

try:
    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    print("Model loaded successfully!")
    print(f"Features: {feature_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Get form data
        data = request.form
        
        # Extract features
        pclass = int(data.get('pclass', 3))
        sex = data.get('sex', 'male')
        age = float(data.get('age', 30))
        fare = float(data.get('fare', 20))
        embarked = data.get('embarked', 'S')
        
        # Encode categorical variables
        sex_encoded = 1 if sex == 'male' else 0
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        embarked_encoded = embarked_map.get(embarked, 2)
        
        # Create feature array in correct order
        features = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex_encoded,
            'Age': age,
            'Fare': fare,
            'Embarked': embarked_encoded
        }])[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'result': 'Survived' if prediction == 1 else 'Did Not Survive',
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'passenger_info': {
                'Passenger Class': pclass,
                'Sex': sex,
                'Age': age,
                'Fare': f"${fare:.2f}",
                'Embarked': embarked
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)