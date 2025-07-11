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
        # Core model features
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
            'power_per_cc', 'torque_per_cc', 'power_to_weight',
            'power_to_torque', 'performance_score'
        ]
        
        # Post-prediction adjustment factors (kept separate from model features)
        self.adjustment_features = [
            'brand_factor', 'mileage_factor', 'age_factor',
            'usage_factor', 'total_depreciation'
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

    def engineer_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = data.copy()
        adjustment_factors = {}
        
        # Define default values for numeric columns
        defaults = {
            'car_age': 0,
            'mileage_num': 0,
            'engine_size_cc_num': 1500,  # Common engine size
            'horse_power_num': 100,      # Average horsepower
            'torque_num': 150,           # Average torque
            'seats_num': 5,              # Standard seating
            'annual_insurance': 40000    # Average insurance
        }
        
        # Ensure numeric columns have no NaN values using sensible defaults
        for col, default_value in defaults.items():
            if col in df.columns:
                if df[col].isna().any():
                    print(f"Warning: Found NaN in {col}, filling with {default_value}")
                    df[col] = df[col].fillna(default_value)
            else:
                print(f"Warning: Missing column {col}, adding with default value {default_value}")
                df[col] = default_value
        
        # Basic numeric transformations with enhanced sensitivity and NaN prevention
        df['car_age_squared'] = df['car_age'] ** 2
        df['mileage_log'] = np.log1p(df['mileage_num'].clip(lower=0))
        df['mileage_per_year'] = df['mileage_num'] / (df['car_age'].clip(lower=1e-6))
        df['engine_size_cc_log'] = np.log1p(df['engine_size_cc_num'].clip(lower=0))
        df['horse_power_log'] = np.log1p(df['horse_power_num'].clip(lower=0))
        df['torque_log'] = np.log1p(df['torque_num'].clip(lower=0))
        
        # Calculate adjustment factors (stored separately)
        adjustment_factors['mileage_factor'] = 0.95 ** (df['mileage_num'].iloc[0] / 10000)
        adjustment_factors['age_factor'] = 0.90 ** df['car_age'].iloc[0]
        
        # Market segment indicators and brand factors
        luxury_makes = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar']
        premium_makes = ['toyota', 'honda', 'volkswagen', 'mazda', 'subaru']
        economy_makes = ['suzuki', 'mitsubishi', 'nissan', 'hyundai', 'kia']
        
        # Calculate brand factor (as adjustment)
        make = df['make_name_cleaned'].iloc[0].lower()
        if make in luxury_makes:
            adjustment_factors['brand_factor'] = 1.3
        elif make in premium_makes:
            adjustment_factors['brand_factor'] = 1.1
        else:
            adjustment_factors['brand_factor'] = 0.9
        
        # Enhanced performance metrics with NaN prevention (model features)
        df['power_per_cc'] = (df['horse_power_num'] / df['engine_size_cc_num'].clip(lower=1)).clip(lower=0, upper=2)
        df['torque_per_cc'] = (df['torque_num'] / df['engine_size_cc_num'].clip(lower=1)).clip(lower=0, upper=10)
        df['power_to_weight'] = (df['horse_power_num'] / (df['engine_size_cc_num'].clip(lower=1) / 500)).clip(lower=0, upper=100)
        df['power_to_torque'] = (df['horse_power_num'] / df['torque_num'].clip(lower=1)).clip(lower=0, upper=5)
        
        # Calculate usage factor (as adjustment)
        if df['usage_type_clean'].iloc[0] == 'Foreign Used':
            adjustment_factors['usage_factor'] = 1.1
        elif df['usage_type_clean'].iloc[0] == 'Kenyan Used':
            adjustment_factors['usage_factor'] = 0.9
        else:
            adjustment_factors['usage_factor'] = 1.0
        
        # Performance score (model feature)
        df['performance_score'] = (df['power_to_weight'] * df['power_to_torque']) ** 0.5
        
        # Calculate total depreciation (as adjustment)
        adjustment_factors['total_depreciation'] = (
            adjustment_factors['age_factor'] * 
            adjustment_factors['mileage_factor'] * 
            adjustment_factors['brand_factor']
        )
        
        return df, adjustment_factors

    def validate_features(self, data: pd.DataFrame) -> List[str]:
        warnings = []
        errors = []
        
        # Check for NaN values first
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            errors.append(f"Missing values found in columns: {', '.join(nan_columns)}")
            
        # Ensure all required numeric columns are present and valid
        required_numeric = {
            'car_age': (0, 50),
            'mileage_num': (0, 1000000),
            'engine_size_cc_num': (600, 8000),
            'horse_power_num': (30, 1000),
            'torque_num': (30, 1200),
            'seats_num': (2, 15)
        }
        
        for col, (min_val, max_val) in required_numeric.items():
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")
            elif data[col].isna().any():
                errors.append(f"Column {col} contains missing values")
            elif (data[col] < min_val).any() or (data[col] > max_val).any():
                errors.append(f"Column {col} contains values outside valid range ({min_val}-{max_val})")
        
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
        
        # Advanced ratio validations with enhanced thresholds
        power_per_cc = power / (engine + 1e-6)
        if power_per_cc > 0.25:  # Adjusted for high-performance vehicles
            warnings.append(f"Power-to-engine ratio ({power_per_cc:.2f} hp/cc) is unusually high")
        elif power_per_cc < 0.03:  # Added lower bound
            warnings.append(f"Power-to-engine ratio ({power_per_cc:.2f} hp/cc) is unusually low")
        
        torque_per_hp = torque / (power + 1e-6)
        if torque_per_hp > 5:  # Adjusted for modern diesel engines
            warnings.append(f"Torque-to-power ratio ({torque_per_hp:.1f} Nm/hp) is unusually high")
        elif torque_per_hp < 0.5:  # Added lower bound
            warnings.append(f"Torque-to-power ratio ({torque_per_hp:.1f} Nm/hp) is unusually low")
        
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
        
        # First ensure no NaN values in numeric columns
        for col in self.numeric_features + self.derived_features + ['annual_insurance']:
            if col in df.columns and df[col].isna().any():
                print(f"Warning: Found NaN in {col} during prefix addition, filling with 0")
                df[col] = df[col].fillna(0)
        
        # Add prefixes
        renamed = {}
        for col in df.columns:
            if col in ['annual_insurance']:
                renamed[col] = f'num_insurance__{col}'
            elif col in self.numeric_features or col in self.derived_features:
                renamed[col] = f'num_main__{col}'
            elif col in self.categorical_features:
                renamed[col] = f'cat__{col}'
        
        # Rename and verify no NaN values
        result = df.rename(columns=renamed)
        
        # Final NaN check
        nan_cols = result.columns[result.isna().any()].tolist()
        if nan_cols:
            print("Warning: NaN values found after prefix addition in:", nan_cols)
            print("Filling remaining NaN values with 0")
            result = result.fillna(0)
            
        return result

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
        
        # Step 2: Engineer features and get adjustment factors
        engineered_data, adjustment_factors = features.engineer_features(input_data)
        debug_dataframe(engineered_data, "After Feature Engineering")
        
        # Step 3: Create one-hot encoded features
        encoded_data = features.create_one_hot_features(engineered_data)
        debug_dataframe(encoded_data, "After One-Hot Encoding")
        
        # Step 4: Add prefixes to numeric features
        numeric_data = features.add_prefix(engineered_data)
        debug_dataframe(numeric_data, "After Adding Prefixes")
        
        # Step 5: Combine all features
        final_data = pd.concat([numeric_data, encoded_data], axis=1)
        
        # Step 6: Ensure all required features are present and handle missing/NaN values
        if hasattr(_current_model, 'feature_names_in_'):
            required_features = set(_current_model.feature_names_in_)
            current_features = set(final_data.columns)
            
            # Handle missing features
            missing_features = required_features - current_features
            if missing_features:
                print(f"Adding missing features: {missing_features}")
                missing_df = pd.DataFrame(
                    0,
                    columns=list(missing_features),
                    index=final_data.index
                )
                final_data = pd.concat([final_data, missing_df], axis=1)
            
            # Ensure correct column order and no NaN values
            final_data = final_data[_current_model.feature_names_in_]
            
            # Check for any remaining NaN values
            nan_cols = final_data.columns[final_data.isna().any()].tolist()
            if nan_cols:
                print(f"Warning: Found NaN values in features: {nan_cols}")
                print("Filling remaining NaN values with 0")
                final_data = final_data.fillna(0)            # Step 7: Debug NaN values
            print("Checking for NaN values before normalization:")
            nan_cols = final_data.columns[final_data.isna().any()].tolist()
            if nan_cols:
                print("NaN found in columns:", nan_cols)
                print("NaN counts:", final_data[nan_cols].isna().sum())
            
            # Fill NaN values with appropriate defaults before normalization
            numeric_cols = [col for col in final_data.columns if col.startswith('num_')]
            for col in numeric_cols:
                if final_data[col].isna().any():
                    if 'log' in col:
                        final_data[col] = final_data[col].fillna(0)  # For log features
                    else:
                        final_data[col] = final_data[col].fillna(final_data[col].mean() if not final_data[col].empty else 0)
                
                # Normalize only if we have valid values
                if final_data[col].std() != 0:
                    final_data[col] = (final_data[col] - final_data[col].mean()) / final_data[col].std()
                    
            # Final NaN check
            if final_data.isna().any().any():
                raise ValueError(f"NaN values found in columns: {final_data.columns[final_data.isna().any()].tolist()}")# Step 8: Make prediction
        prediction = _current_model.predict(final_data)
        
        # Step 9: Apply market adjustments
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction = np.exp(prediction[0])
            
            # Apply all adjustment factors
            total_adjustment = (
                adjustment_factors['brand_factor'] *
                adjustment_factors['mileage_factor'] *
                adjustment_factors['age_factor'] *
                adjustment_factors['usage_factor']
            )
            
            prediction *= total_adjustment
            
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

def debug_dataframe(df: pd.DataFrame, stage: str) -> None:
    """Helper function to debug DataFrame issues"""
    print(f"\nDebugging DataFrame at stage: {stage}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"NaN values found in columns: {nan_cols}")
        for col in nan_cols:
            print(f"Column {col}: {df[col].isna().sum()} NaN values")
    
    # Check for all-zero columns
    zero_cols = df.columns[(df == 0).all()].tolist()
    if zero_cols:
        print(f"Columns with all zeros: {zero_cols}")
    
    # Print basic statistics for numeric columns
    print("\nNumeric columns summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
