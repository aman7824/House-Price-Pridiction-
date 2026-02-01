"""
Flask REST API for Indian House Price Prediction.

Endpoints:
    GET /         - Serve frontend
    POST /predict - Predict house price
    GET /health   - Health check
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import predict_price

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')


@app.route('/')
def index():
    """Serve the frontend."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Indian House Price Prediction API'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house price from input features.
    
    Expected JSON body:
    {
        "bhk": 2,
        "area_sqft": 1200,
        "city": "Bangalore",
        "locality": "Whitefield",
        "age": 5,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "parking": 1,
        "lift": 1
    }
    
    Returns:
    {
        "predicted_price_inr": 4500000,
        "formatted": "₹ 45.00 L",
        "lower": 4200000,
        "upper": 4800000,
        "lower_formatted": "₹ 42.00 L",
        "upper_formatted": "₹ 48.00 L",
        "model": "XGBoost",
        "rmse": 350000,
        "mae": 280000,
        "r2": 0.85
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = ['bhk', 'area_sqft', 'city', 'locality', 'age', 
                          'floor', 'bathrooms', 'balconies', 'parking', 'lift']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate numeric fields
        numeric_fields = ['bhk', 'area_sqft', 'age', 'floor', 'bathrooms', 
                         'balconies', 'parking', 'lift']
        
        for field in numeric_fields:
            try:
                data[field] = int(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Field "{field}" must be a valid number'
                }), 400
        
        # Validate ranges
        if data['bhk'] < 1 or data['bhk'] > 10:
            return jsonify({'error': 'BHK must be between 1 and 10'}), 400
        
        if data['area_sqft'] < 100 or data['area_sqft'] > 10000:
            return jsonify({'error': 'Area must be between 100 and 10,000 sqft'}), 400
        
        if data['age'] < 0 or data['age'] > 100:
            return jsonify({'error': 'Age must be between 0 and 100 years'}), 400
        
        if data['floor'] < 0 or data['floor'] > 50:
            return jsonify({'error': 'Floor must be between 0 and 50'}), 400
        
        if data['bathrooms'] < 1 or data['bathrooms'] > 10:
            return jsonify({'error': 'Bathrooms must be between 1 and 10'}), 400
        
        if data['balconies'] < 0 or data['balconies'] > 10:
            return jsonify({'error': 'Balconies must be between 0 and 10'}), 400
        
        if data['parking'] not in [0, 1, 2, 3, 4]:
            return jsonify({'error': 'Parking must be 0, 1, 2, 3, or 4'}), 400
        
        if data['lift'] not in [0, 1]:
            return jsonify({'error': 'Lift must be 0 or 1'}), 400
        
        # Check if model exists
        model_path = "models/pipeline.joblib"
        if not os.path.exists(model_path):
            return jsonify({
                'error': 'Model not found. Please train the model first using: python train.py'
            }), 500
        
        # Make prediction
        result = predict_price(data, model_path)
        
        return jsonify(result), 200
        
    except FileNotFoundError:
        return jsonify({
            'error': 'Model file not found. Please train the model first.'
        }), 500
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('models/pipeline.joblib'):
        print("\n" + "=" * 70)
        print("WARNING: Model file not found!")
        print("=" * 70)
        print("Please train the model first by running:")
        print("  python train.py")
        print("=" * 70 + "\n")
    
    # Run Flask app
    print("\n" + "=" * 70)
    print("Starting Indian House Price Prediction API")
    print("=" * 70)
    print("API will be available at: http://0.0.0.0:5000")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
