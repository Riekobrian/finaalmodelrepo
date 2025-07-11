# Car Price Predictor

A machine learning web application that predicts car prices based on various features.

## Features
- Advanced price prediction using a Stacking Regressor model
- Interactive web interface with real-time predictions
- Support for multiple car makes and models
- Feature sensitivity analysis
- Price impact analysis

## Local Development
1. Clone the repository
```bash
git clone https://github.com/Riekobrian/finaalmodelrepo.git
cd finaalmodelrepo
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## Features Used for Prediction
- Make and model
- Car age and mileage
- Engine specifications (size, power, torque)
- Body type and transmission
- Usage type and condition
- And more...

## Model Details
The app uses a Stacking Regressor that combines multiple base models for better prediction accuracy. The model is automatically downloaded when the app starts.

## Deployment
The app is deployed on Streamlit Cloud and can be accessed at: [Your-App-URL]
