import os
import glob
from joblib import load
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Union

# Dropbox URLs for model artifacts
MODEL_URL = "https://www.dropbox.com/scl/fi/6f1y7jfycbiry4zk4mdtc/20250521_165326_StackingRegressor_Final.joblib?rlkey=ptba91lcfxmn7dmjlj0qcqs0a&dl=1"
PREPROCESSOR_URL = "https://www.dropbox.com/scl/fi/9y53zgq9qk6k6yet96rzt/preprocessor.joblib?rlkey=aue8lzvsgt0wdyvxrzyzw4n1p&dl=1"

def download_artifact(url: str, filename: str) -> str:
    """Download a model artifact from Dropbox"""
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded successfully")
    return file_path

class ModelFeatures:
    def __init__(self):
        self.numeric_features = [
            'annual_insurance', 'car_age', 'mileage_num', 'engine_size_cc_num',
            'horse_power_num', 'torque_num', 'seats_num'
        ]
        self.categorical_features = [
            'fuel_type_cleaned', 'transmission_cleaned', 'drive_type_cleaned',
            'usage_type_clean', 'body_type_cleaned', 'make_name_cleaned',
            'model_name_cleaned'
        ]
        self.derived_features = [
            'car_age_squared', 'mileage_log', 'mileage_per_year',
            'engine_size_cc_log', 'horse_power_log', 'torque_log',
            'power_per_cc', 'mileage_per_cc', 'is_luxury_make'
        ]
        
        self.known_categories = {
            'fuel_type_cleaned': ['petrol', 'diesel', 'hybrid_petrol', 'hybrid_diesel', 
                                'electric', 'plugin_hybrid_petrol', 'unknown'],
            'transmission_cleaned': ['automatic', 'manual', 'automated_manual', 'unknown'],
            'drive_type_cleaned': ['2wd', '2wd_front', '2wd_mid_engine', '2wd_rear', 
                                 '4wd', 'awd', 'unknown'],
            'body_type_cleaned': ['sedan', 'suv', 'hatchback', 'wagon', 'van_minivan', 
                                'pickup_truck', 'coupe', 'bus', 'convertible']
        }

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Basic numeric transformations with enhanced sensitivity
        df['car_age_squared'] = df['car_age'] ** 2
        df['mileage_log'] = np.log1p(df['mileage_num'])
        df['mileage_per_year'] = df['mileage_num'] / (df['car_age'] + 1e-6)
        df['engine_size_cc_log'] = np.log1p(df['engine_size_cc_num'])
        df['horse_power_log'] = np.log1p(df['horse_power_num'])
        df['torque_log'] = np.log1p(df['torque_num'])
        
        # Enhanced depreciation factors (from insights)
        df['mileage_factor'] = 0.95 ** (df['mileage_num'] / 10000)  # More gradual mileage impact
        df['age_factor'] = 0.90 ** df['car_age']  # Steeper age depreciation
        
        # Enhanced performance metrics with better ratios
        df['power_per_cc'] = df['horse_power_num'] / (df['engine_size_cc_num'] + 1e-6)
        df['torque_per_cc'] = df['torque_num'] / (df['engine_size_cc_num'] + 1e-6)
        df['power_to_weight'] = df['horse_power_num'] / (df['engine_size_cc_num'] / 500)
        df['power_to_torque'] = df['horse_power_num'] / (df['torque_num'] + 1e-6)
        
        # Market segment indicators
        luxury_makes = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar']
        premium_makes = ['toyota', 'honda', 'volkswagen', 'mazda', 'subaru']
        economy_makes = ['suzuki', 'mitsubishi', 'nissan', 'hyundai', 'kia']
        
        # Brand value factors
        df['brand_factor'] = 1.0  # Base factor
        df.loc[df['make_name_cleaned'].isin(luxury_makes), 'brand_factor'] = 1.3  # Luxury premium
        df.loc[df['make_name_cleaned'].isin(premium_makes), 'brand_factor'] = 1.1  # Premium boost
        df.loc[df['make_name_cleaned'].isin(economy_makes), 'brand_factor'] = 0.9  # Economy adjustment
        
        # Progressive depreciation factors
        df['mileage_factor'] = 0.95 ** (df['mileage_num'] / 10000)  # More gradual mileage impact
        df['age_factor'] = 0.90 ** df['car_age']  # Steeper age depreciation
        
        # Enhanced brand value factors with more granular categories
        luxury_makes = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar']
        premium_makes = ['toyota', 'honda', 'volkswagen', 'mazda', 'subaru']
        economy_makes = ['suzuki', 'mitsubishi', 'nissan', 'hyundai', 'kia']
        
        df['brand_factor'] = 1.0  # Base factor
        df.loc[df['make_name_cleaned'].isin(luxury_makes), 'brand_factor'] = 1.3
        df.loc[df['make_name_cleaned'].isin(premium_makes), 'brand_factor'] = 1.1
        df.loc[df['make_name_cleaned'].isin(economy_makes), 'brand_factor'] = 0.9
        
        # Enhanced depreciation calculations
        df['total_depreciation'] = df['age_factor'] * df['mileage_factor'] * df['brand_factor']
        
        # Usage type impact with more granular effects
        df['usage_factor'] = 1.0  # Base factor
        df.loc[df['usage_type_clean'] == 'Foreign Used', 'usage_factor'] = 1.1
        df.loc[df['usage_type_clean'] == 'Kenyan Used', 'usage_factor'] = 0.9
        
        # Sophisticated interaction features
        df['luxury_age_impact'] = df['brand_factor'] * df['age_factor']
        df['performance_score'] = (df['power_to_weight'] * df['power_to_torque']) ** 0.5
        df['condition_score'] = df['total_depreciation'] * df['usage_factor']
        
        # Market segment interactions
        df['market_segment_value'] = df['brand_factor'] * df['performance_score'] * df['condition_score']
        
        return df

    def validate_features(self, data: pd.DataFrame) -> List[str]:
        warnings = []
        errors = []
        
        # Age validation with hard limits
        age = data['car_age'].iloc[0]
        if age > 50:
            errors.append("Car age exceeds maximum allowed (50 years)")
        elif age > 30:
            warnings.append("Car age is over 30 years - prediction may be less accurate")
        elif age > 20:
            warnings.append("Car is over 20 years old - consider condition carefully")
        elif age < 0:
            errors.append("Car age cannot be negative")
        
        # Mileage validation with realistic bounds
        mileage = data['mileage_num'].iloc[0]
        if mileage > 1000000:
            errors.append("Mileage exceeds reasonable limit (1,000,000 km)")
        elif mileage > 500000:
            warnings.append("Mileage is unusually high - verify accuracy")
        elif mileage > 300000:
            warnings.append("High mileage may affect price significantly")
        elif mileage < 0:
            errors.append("Mileage cannot be negative")
        
        # Engine size validation with make-specific ranges
        engine = data['engine_size_cc_num'].iloc[0]
        make = data['make_name_cleaned'].iloc[0]
        
        # Make-specific engine size validation
        if make in ['bmw', 'mercedes', 'audi']:
            if engine > 6000:
                warnings.append(f"Engine size {engine}cc is unusually large for {make}")
            elif engine < 1200:
                warnings.append(f"Engine size {engine}cc is unusually small for {make}")
        else:
            if engine > 8000:
                errors.append("Engine size exceeds maximum allowed (8,000 cc)")
            elif engine < 600:
                errors.append("Engine size below minimum allowed (600 cc)")
        
        # Power validation with make-specific ranges
        power = data['horse_power_num'].iloc[0]
        if make in ['porsche', 'ferrari', 'lamborghini']:
            if power > 800:
                warnings.append(f"Very high horsepower ({power}hp) - verify specifications")
            elif power < 300:
                warnings.append(f"Unusually low horsepower ({power}hp) for {make}")
        else:
            if power > 500:
                warnings.append(f"Very high horsepower ({power}hp) - verify specifications")
            elif power < 30:
                errors.append("Horsepower below minimum allowed (30 hp)")
        
        # Torque validation
        torque = data['torque_num'].iloc[0]
        if torque > 1000:
            warnings.append(f"Very high torque ({torque}Nm) - verify specifications")
        elif torque < 30:
            errors.append("Torque below minimum allowed (30 Nm)")
        
        # Advanced ratio validations
        power_per_cc = power / (engine + 1e-6)
        if power_per_cc > 0.2:
            warnings.append(f"Power-to-engine ratio ({power_per_cc:.2f} hp/cc) is unusually high")
        
        torque_per_hp = torque / (power + 1e-6)
        if torque_per_hp > 4:
            warnings.append(f"Torque-to-power ratio ({torque_per_hp:.1f} Nm/hp) is unusually high")
        
        # Insurance validation
        insurance = data['annual_insurance'].iloc[0]
        if insurance > 500000:
            warnings.append("Annual insurance cost is unusually high")
        elif insurance < 10000:
            warnings.append("Annual insurance cost seems unusually low")
        
        # Raise error if any hard limits are violated
        if errors:
            raise ValueError("Validation errors: " + "; ".join(errors))
            
        return warnings
            
        return warnings

    def add_prefix(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        renamed = {}
        
        for col in df.columns:
            if col in ['annual_insurance']:
                renamed[col] = f'num_insurance__{col}'
            elif col in self.numeric_features or col in self.derived_features:
                renamed[col] = f'num_main__{col}'
            elif col in self.categorical_features:
                renamed[col] = f'cat__{col}'
                
        return df.rename(columns=renamed)

    def create_one_hot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        one_hot_data = {}
        
        for feature in self.categorical_features:
            if feature in data.columns:
                value = data[feature].iloc[0]
                categories = self.known_categories.get(feature, [value])
                for category in categories:
                    col_name = f'cat__{feature}_{category}'
                    one_hot_data[col_name] = [1 if value == category else 0]
        
        return pd.DataFrame(one_hot_data, index=data.index)

# Global variables to store current model state
_current_model = None
_current_preprocessor = None

def load_model():
    """Load the model from cloud storage"""
    global _current_model
    
    if _current_model is None:
        try:
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Download and load model
            model_path = download_artifact(MODEL_URL, "model.joblib")
            print("Model downloaded, attempting to load...")
            _current_model = load(model_path)
            print("Model loaded successfully")
            
        except Exception as e:
            import traceback
            print(f"Error loading model: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            raise ValueError(f"Failed to load model: {str(e)}")

def predict_price(input_data: Union[Dict, pd.DataFrame]) -> float:
    """Make a prediction using the current model with proper feature handling"""
    global _current_model
    
    # Load model if not already loaded
    if _current_model is None:
        load_model()
        
    # Convert dictionary to DataFrame if needed
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    try:
        features = ModelFeatures()
        
        # Step 1: Validate input
        warnings = features.validate_features(input_data)
        for warning in warnings:
            print(f"Warning: {warning}")
        
        # Step 2: Engineer features
        engineered_data = features.engineer_features(input_data)
        
        # Step 3: Create one-hot encoded features
        encoded_data = features.create_one_hot_features(engineered_data)
        
        # Step 4: Add prefixes to numeric features
        numeric_data = features.add_prefix(engineered_data)
        
        # Step 5: Combine all features
        final_data = pd.concat([numeric_data, encoded_data], axis=1)
        
        # Step 6: Ensure all required features are present
        if hasattr(_current_model, 'feature_names_in_'):
            required_features = set(_current_model.feature_names_in_)
            current_features = set(final_data.columns)
            
            missing_features = required_features - current_features
            if missing_features:
                missing_df = pd.DataFrame(
                    0,
                    columns=list(missing_features),
                    index=final_data.index
                )
                final_data = pd.concat([final_data, missing_df], axis=1)
            
            final_data = final_data[_current_model.feature_names_in_]
        
        # Step 7: Make prediction
        prediction = _current_model.predict(final_data)
        
        # Step 8: Convert log prediction if needed
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction = np.exp(prediction[0])
            
        return prediction
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def get_model_info() -> dict:
    """Get information about the current model"""
    if _current_model is None:
        load_model()
    
    return {
        "model_type": str(_current_model.__class__.__name__),
        "features": list(_current_model.feature_names_in_) if hasattr(_current_model, 'feature_names_in_') else None
    }
