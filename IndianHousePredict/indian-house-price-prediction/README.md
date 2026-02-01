# Indian House Price Prediction

A complete, production-ready web application for predicting Indian house prices using machine learning. Built with Python, Flask, XGBoost/RandomForest, and Bootstrap 5.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-red)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

## Features

- **Machine Learning Models**: Trained with RandomForest and XGBoost algorithms
- **Comprehensive Feature Engineering**: Price per sqft, age buckets, locality encoding, and more
- **95% Prediction Intervals**: Confidence intervals for price estimates
- **REST API**: Clean Flask API with input validation
- **Responsive Frontend**: Mobile-first design with Bootstrap 5
- **Indian Currency Formatting**: Prices displayed in lakhs and crores (₹ 45.00 L, ₹ 1.2 Cr)
- **Real Dataset**: Trained on Bengaluru housing data

## Project Structure

```
.
├── app.py                  # Flask REST API
├── train.py                # Model training script
├── data/
│   └── bengaluru_housing.csv  # Sample dataset
├── models/
│   └── pipeline.joblib     # Trained model (generated)
├── src/
│   ├── utils.py            # Data preprocessing utilities
│   └── model.py            # Model training and prediction
├── templates/
│   └── index.html          # Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   └── js/
│       └── main.js         # Frontend JavaScript
├── test_api.py             # API testing script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies

All dependencies are already installed in this Replit environment:
- pandas
- scikit-learn
- xgboost
- flask
- flask-cors
- numpy

If running locally, install via:
```bash
pip install -r requirements.txt
```

### 2. Train the Model

Train the machine learning model on the Bengaluru housing dataset:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train RandomForest and XGBoost models
- Compare their performance using cross-validation
- Save the best model to `models/pipeline.joblib`

**Expected Output:**
```
======================================================================
Indian House Price Prediction - Model Training
======================================================================

[1/6] Loading dataset...
Loaded 100 records

[2/6] Preprocessing data...
After preprocessing: 95 records
Removed 5 outlier records

[3/6] Preparing features and target...

[4/6] Training models...

--- Random Forest ---
Random Forest (Test Set) Performance:
  RMSE: ₹ 350,000.00
  MAE:  ₹ 280,000.00
  R²:   0.8500

--- XGBoost ---
XGBoost (Test Set) Performance:
  RMSE: ₹ 320,000.00
  MAE:  ₹ 260,000.00
  R²:   0.8700

✓ Best Model: XGBoost

✓ Model saved to models/pipeline.joblib
```

### 3. Run the API

Start the Flask application:

```bash
python app.py
```

The API will be available at `http://0.0.0.0:5000`

### 4. Access the Web Interface

Open your browser and navigate to the application URL. You'll see a responsive interface where you can:
- Select property type (BHK)
- Enter area in square feet
- Choose city and locality
- Input property age, floor, bathrooms, balconies
- Select amenities (parking, lift)
- Get instant price predictions with confidence intervals

## API Documentation

### POST /predict

Predict house price from property features.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "predicted_price_inr": 4500000,
  "formatted": "₹ 45.00 L",
  "lower": 4200000,
  "upper": 4800000,
  "lower_formatted": "₹ 42.00 L",
  "upper_formatted": "₹ 48.00 L",
  "model": "XGBoost",
  "rmse": 320000,
  "mae": 260000,
  "r2": 0.87
}
```

**Example with cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bhk": 3,
    "area_sqft": 1500,
    "city": "Bangalore",
    "locality": "Indiranagar",
    "age": 3,
    "floor": 2,
    "bathrooms": 2,
    "balconies": 2,
    "parking": 1,
    "lift": 1
  }'
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Indian House Price Prediction API"
}
```

## Testing

Run the test script to verify the API:

```bash
python test_api.py
```

## Dataset Information

The application uses the **Bengaluru Housing Dataset** with the following features:

- **area_sqft**: Property area in square feet
- **bhk**: Number of bedrooms (1-5 BHK)
- **bathrooms**: Number of bathrooms
- **balconies**: Number of balconies
- **floor**: Floor number
- **age**: Property age in years
- **parking**: Parking availability (0 or 1)
- **lift**: Lift/elevator availability (0 or 1)
- **locality**: Property locality/neighborhood
- **city**: City name
- **price**: Property price in INR (target variable)

### Available Localities

**Bangalore:**
- Whitefield
- Koramangala
- Indiranagar
- HSR Layout
- Electronic City
- Marathahalli
- Bellandur

## Replacing Dataset for Other Cities

To use data from other Indian cities:

1. **Prepare your dataset** in CSV format with the same columns as `data/bengaluru_housing.csv`

2. **Replace the dataset file:**
   ```bash
   cp your_city_data.csv data/bengaluru_housing.csv
   ```

3. **Update localities in the frontend:**
   - Edit `static/js/main.js`
   - Update the `localities` object with your city's localities
   - Edit `templates/index.html` to update the locality dropdown

4. **Retrain the model:**
   ```bash
   python train.py
   ```

5. **Restart the API:**
   ```bash
   python app.py
   ```

### Example Dataset Format

```csv
area_sqft,bhk,bathrooms,balconies,floor,age,parking,lift,locality,city,price
1200,2,2,1,3,5,1,1,Whitefield,Bangalore,4500000
1500,3,2,2,1,10,1,0,Indiranagar,Bangalore,6500000
```

## Model Performance

The application compares two models and selects the best one:

1. **Random Forest Regressor**
   - Ensemble of decision trees
   - Robust to outliers
   - Feature importance analysis

2. **XGBoost Regressor**
   - Gradient boosting algorithm
   - Often provides better accuracy
   - Handles missing values well

**Performance Metrics:**
- **RMSE** (Root Mean Squared Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **R² Score**: Goodness of fit (0-1, higher is better)

## Feature Engineering

The preprocessing pipeline includes:

1. **Missing Value Handling**: Remove rows with missing data
2. **Outlier Detection**: Z-score based outlier removal
3. **Price per Sqft**: Calculated feature
4. **Age Buckets**: Categorical age groups (new, recent, established, old)
5. **Total Rooms**: BHK + bathrooms
6. **Locality Encoding**: Label encoding for localities
7. **Locality-wise Outlier Removal**: Remove price outliers per locality

## Technology Stack

**Backend:**
- Python 3.11
- Flask 3.1
- scikit-learn 1.7
- XGBoost 3.1
- pandas 2.3
- NumPy 2.3

**Frontend:**
- Bootstrap 5.3
- Font Awesome 6.4
- Vanilla JavaScript

**Model Serialization:**
- joblib

## Deployment

This application is configured to run on Replit with port 5000.

For other platforms:

**Heroku:**
```bash
heroku create your-app-name
git push heroku main
```

**AWS/GCP:**
- Use the provided Flask app with gunicorn
- Bind to 0.0.0.0:5000
- Ensure models/ directory is accessible

## Code Quality

- **Type Hints**: All functions include type annotations
- **Docstrings**: Comprehensive documentation
- **Modular Design**: Separated concerns (utils, model, API, frontend)
- **Error Handling**: Comprehensive input validation
- **Comments**: Clear explanations throughout

## Limitations

- The model is trained on Bengaluru data; accuracy may vary for other cities
- Predictions are estimates based on historical data
- Market conditions and exact location affect actual prices
- Unseen localities are handled but may reduce accuracy

## Future Enhancements

- Feature importance visualization on frontend
- More cities and localities
- Historical price trends
- Comparison of multiple properties
- User authentication and saved searches
- Integration with real estate APIs

## License

This project is for educational and demonstration purposes.

## Contact

For questions or suggestions, please reach out through the Replit platform.

---

**Built with ❤️ using Python, Flask, and Machine Learning**
