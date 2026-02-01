# Indian House Price Prediction Application

## Overview
A complete, production-ready web application for predicting Indian house prices using machine learning. Built with Python, Flask, XGBoost/RandomForest, and Bootstrap 5.

**Current Status:** Fully functional and deployed
**Last Updated:** October 29, 2025

## Project Architecture

### Backend (Python/Flask)
- **app.py**: Flask REST API with `/predict` and `/health` endpoints
- **train.py**: Model training script that compares RandomForest vs XGBoost
- **src/model.py**: Model training, evaluation, and prediction functions
- **src/utils.py**: Data preprocessing and feature engineering utilities

### Frontend (Bootstrap 5 + Vanilla JS)
- **templates/index.html**: Responsive single-page application
- **static/css/style.css**: Custom styling with gradient backgrounds
- **static/js/main.js**: Form handling, API calls, and result display

### Data & Models
- **data/bengaluru_housing.csv**: Sample dataset with 96 records
- **models/pipeline.joblib**: Trained XGBoost model (R² = 0.9824)

## Key Features

1. **Machine Learning**
   - XGBoost model with 98.24% R² score
   - 95% prediction intervals for confidence bounds
   - RMSE: ₹ 472,139.85
   - MAE: ₹ 286,412.59

2. **Feature Engineering**
   - Price per square foot calculation
   - Age buckets (new, recent, established, old)
   - Locality-based outlier removal
   - Label encoding for categorical variables
   - Total rooms (BHK + bathrooms)

3. **API**
   - Input validation with detailed error messages
   - JSON responses with formatted INR (lakhs/crores)
   - Confidence intervals for predictions
   - Model performance metrics included

4. **Frontend**
   - Mobile-first responsive design
   - Real-time form validation
   - Loading states and error handling
   - Indian currency formatting (₹ 45.00 L, ₹ 1.2 Cr)
   - Bootstrap 5 with custom gradients

## Technology Stack

- **Backend**: Python 3.11, Flask 3.1, scikit-learn 1.7, XGBoost 3.1
- **Frontend**: Bootstrap 5.3, Font Awesome 6.4, Vanilla JavaScript
- **ML Libraries**: pandas 2.3, numpy 2.3, joblib 1.5
- **Deployment**: Replit (port 5000)

## Recent Changes

### October 29, 2025 - Initial Release
- Created complete application from scratch
- Trained XGBoost model achieving 98.24% R² score
- Implemented responsive frontend with Bootstrap 5
- Added comprehensive documentation and test suite
- Configured Flask workflow for continuous deployment

## Dataset Information

**Source**: Bengaluru Housing Data (sample dataset included)
**Records**: 96 properties across 7 localities
**Features**: 11 input features + 1 target (price)

**Localities**:
- Whitefield
- Koramangala
- Indiranagar
- HSR Layout
- Electronic City
- Marathahalli
- Bellandur

## Usage Instructions

### Training the Model
```bash
python train.py
```
This trains both RandomForest and XGBoost, compares performance, and saves the best model.

### Running the Application
The Flask server is automatically started via the workflow. Access the web interface through the Replit webview.

### Testing the API
```bash
python test_api.py
```
Runs comprehensive API tests with sample predictions.

### Replacing Dataset for Other Cities
1. Replace `data/bengaluru_housing.csv` with new city data (same format)
2. Update localities in `static/js/main.js` and `templates/index.html`
3. Retrain model: `python train.py`
4. Restart Flask server

## File Structure
```
.
├── app.py                      # Flask REST API
├── train.py                    # Model training script
├── test_api.py                 # API testing suite
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── data/
│   └── bengaluru_housing.csv  # Sample dataset
├── models/
│   └── pipeline.joblib        # Trained model
├── src/
│   ├── utils.py               # Preprocessing utilities
│   └── model.py               # ML model functions
├── templates/
│   └── index.html             # Frontend HTML
└── static/
    ├── css/
    │   └── style.css          # Custom styles
    └── js/
        └── main.js            # Frontend JavaScript
```

## Performance Metrics

**XGBoost Model (Selected)**:
- Test Set RMSE: ₹ 472,139.85
- Test Set MAE: ₹ 286,412.59
- Test Set R²: 0.9824
- CV RMSE: ₹ 325,715.42 (±₹ 99,490.35)

**Top Features by Importance**:
1. BHK (46.22%)
2. Parking (20.15%)
3. Total Rooms (16.32%)
4. Area (9.89%)
5. Balconies (5.12%)

## User Preferences

None set yet - project just created.

## Known Limitations

1. Model trained on Bengaluru data only - accuracy may vary for other cities
2. Predictions are estimates based on historical data
3. Unseen localities handled but may reduce accuracy slightly
4. Development server (Flask) - production deployment should use gunicorn

## Future Enhancements

1. Feature importance visualization on frontend
2. Support for more Indian cities
3. Historical price trends
4. Property comparison features
5. User authentication and saved searches
6. Integration with real estate APIs
7. Production WSGI server configuration

## API Endpoints

### POST /predict
Predicts house price from property features.

**Request**:
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

**Response**:
```json
{
  "predicted_price_inr": 4500000,
  "formatted": "₹ 45.00 L",
  "lower": 4200000,
  "upper": 4800000,
  "lower_formatted": "₹ 42.00 L",
  "upper_formatted": "₹ 48.00 L",
  "model": "XGBoost",
  "rmse": 472139.85,
  "mae": 286412.59,
  "r2": 0.9824
}
```

### GET /health
Health check endpoint.

## Deployment

Currently deployed on Replit with Flask development server on port 5000. The application is configured with:
- Workflow: Flask Server (python app.py)
- Port: 5000
- Output: webview
- Status: RUNNING

Ready for production deployment to platforms like Heroku, AWS, or GCP with minimal configuration changes.
