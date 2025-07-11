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
            'horse_power_num', 'torque_num', 'seats_num', 'acceleration_num',
            'is_luxury_make'
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
                                'pickup_truck', 'coupe', 'bus', 'convertible'],
            'make_name_cleaned': [
                'toyota', 'honda', 'mazda', 'nissan', 'suzuki', 'mitsubishi', 'subaru',  # Japanese
                'bmw', 'mercedes', 'audi', 'volkswagen', 'porsche',  # German
                'lexus', 'land rover', 'jaguar',  # Luxury
                'hyundai', 'kia',  # Korean
                'other'  # Fallback
            ]
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
            'annual_insurance': 40000,   # Average insurance
            'acceleration_num': 12.0,    # Average acceleration 0-100km/h
            'is_luxury_make': 0          # Default to non-luxury
        }
        
        # Ensure numeric columns exist and have no NaN values
        for col, default_value in defaults.items():
            if col not in df.columns:
                print(f"Warning: Missing column {col}, adding with default value {default_value}")
                df[col] = default_value
            elif df[col].isna().any():
                print(f"Warning: Found NaN in {col}, filling with {default_value}")
                df[col] = df[col].fillna(default_value)
            
            # Ensure values are numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_value)
        
        # Basic numeric transformations with enhanced sensitivity and NaN prevention
        df['car_age_squared'] = df['car_age'].clip(lower=0) ** 2
        df['mileage_log'] = np.log1p(df['mileage_num'].clip(lower=0))
        df['mileage_per_year'] = df['mileage_num'] / df['car_age'].clip(lower=1)
        df['engine_size_cc_log'] = np.log1p(df['engine_size_cc_num'].clip(lower=0))
        df['horse_power_log'] = np.log1p(df['horse_power_num'].clip(lower=0))
        df['torque_log'] = np.log1p(df['torque_num'].clip(lower=0))
        
        # Enhanced performance metrics with NaN prevention
        # Weight estimation based on engine size and vehicle type (rough approximation)
        estimated_weight = df['engine_size_cc_num'] * 0.9  # rough estimation of vehicle weight in kg
        df['power_to_weight'] = (df['horse_power_num'] / estimated_weight.clip(lower=1)).clip(lower=0, upper=1)
        df['power_per_cc'] = (df['horse_power_num'] / df['engine_size_cc_num'].clip(lower=1)).clip(lower=0, upper=2)
        df['torque_per_cc'] = (df['torque_num'] / df['engine_size_cc_num'].clip(lower=1)).clip(lower=0, upper=10)
        df['power_to_torque'] = (df['horse_power_num'] / df['torque_num'].clip(lower=1)).clip(lower=0, upper=5)
        df['mileage_per_cc'] = (df['mileage_num'] / df['engine_size_cc_num'].clip(lower=1)).clip(lower=0)
        
        # Calculate adjustment factors (stored separately)
        adjustment_factors['mileage_factor'] = 0.95 ** (df['mileage_num'].iloc[0] / 10000)
        adjustment_factors['age_factor'] = 0.90 ** df['car_age'].iloc[0]
        
        # Market segment indicators and brand factors
        luxury_makes = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar']
        premium_makes = ['toyota', 'honda', 'volkswagen', 'mazda', 'subaru']
        economy_makes = ['suzuki', 'mitsubishi', 'nissan', 'hyundai', 'kia']
        
        # Calculate brand factor (as adjustment)
        make = str(df['make_name_cleaned'].iloc[0]).lower()
        if make in luxury_makes:
            adjustment_factors['brand_factor'] = 1.3
            df['is_luxury_make'] = 1
        elif make in premium_makes:
            adjustment_factors['brand_factor'] = 1.1
            df['is_luxury_make'] = 0
        else:
            adjustment_factors['brand_factor'] = 0.9
            df['is_luxury_make'] = 0
        
        # Calculate usage factor (as adjustment)
        usage = str(df['usage_type_clean'].iloc[0])
        if usage == 'Foreign Used':
            adjustment_factors['usage_factor'] = 1.1
        elif usage == 'Kenyan Used':
            adjustment_factors['usage_factor'] = 0.9
        else:
            adjustment_factors['usage_factor'] = 1.0
        
        # Performance score (model feature)
        df['performance_score'] = (df['power_to_weight'] * df['power_to_torque']).clip(lower=0) ** 0.5
        
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
        
        # Ensure all required numeric columns are present with valid values
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
            else:
                # Convert to numeric if needed
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().any():
                    errors.append(f"Invalid numeric value in column {col}")
                elif (data[col] < min_val).any() or (data[col] > max_val).any():
                    warnings.append(f"Column {col} contains values outside recommended range ({min_val}-{max_val})")
        
        # Validate categorical features
        for feature, valid_categories in self.known_categories.items():
            if feature not in data.columns:
                errors.append(f"Missing required categorical feature: {feature}")
            else:
                value = str(data[feature].iloc[0]).lower()
                if value not in valid_categories:
                    if feature == 'make_name_cleaned':
                        print(f"Warning: Unknown make '{value}', will be treated as 'other'")
                        data.loc[data.index[0], feature] = 'other'
                    else:
                        warnings.append(f"Unknown {feature} value: {value}")
        
        # Make-specific validations
        if 'make_name_cleaned' in data.columns and not data['make_name_cleaned'].isna().any():
            make = str(data['make_name_cleaned'].iloc[0]).lower()
            
            # Engine size validation
            if 'engine_size_cc_num' in data.columns and not data['engine_size_cc_num'].isna().any():
                engine = data['engine_size_cc_num'].iloc[0]
                if make in ['bmw', 'mercedes', 'audi']:
                    if engine > 6000:
                        warnings.append(f"Engine size {engine}cc is unusually large for {make}")
                    elif engine < 1200:
                        warnings.append(f"Engine size {engine}cc is unusually small for {make}")
            
            # Power validation
            if 'horse_power_num' in data.columns and not data['horse_power_num'].isna().any():
                power = data['horse_power_num'].iloc[0]
                if make in ['porsche', 'ferrari', 'lamborghini']:
                    if power > 800:
                        warnings.append(f"Very high horsepower ({power}hp) - verify specifications")
                    elif power < 300:
                        warnings.append(f"Unusually low horsepower ({power}hp) for {make}")
                
                # Power per cc validation if both values are available
                if 'engine_size_cc_num' in data.columns and not data['engine_size_cc_num'].isna().any():
                    engine = data['engine_size_cc_num'].iloc[0]
                    if engine > 0:
                        power_per_cc = power / engine
                        if power_per_cc > 0.25:
                            warnings.append(f"Power-to-engine ratio ({power_per_cc:.2f} hp/cc) is unusually high")
                        elif power_per_cc < 0.03:
                            warnings.append(f"Power-to-engine ratio ({power_per_cc:.2f} hp/cc) is unusually low")
            
            # Torque validation
            if 'torque_num' in data.columns and not data['torque_num'].isna().any():
                torque = data['torque_num'].iloc[0]
                if torque > 1000:
                    warnings.append(f"Very high torque ({torque}Nm) - verify specifications")
        
        # Insurance validation
        if 'annual_insurance' in data.columns and not data['annual_insurance'].isna().any():
            insurance = data['annual_insurance'].iloc[0]
            if insurance > 500000:
                warnings.append("Annual insurance cost is unusually high")
            elif insurance < 10000:
                warnings.append("Annual insurance cost seems unusually low")
        
        if errors:
            raise ValueError("Validation errors: " + "; ".join(errors))
            
        return warnings
            
        return warnings

    def add_prefix(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Define default values for numeric features
        defaults = {
            'annual_insurance': 40000,
            'car_age': 0,
            'mileage_num': 0,
            'engine_size_cc_num': 1500,
            'horse_power_num': 100,
            'torque_num': 150,
            'seats_num': 5,
            'car_age_squared': 0,
            'mileage_log': 0,
            'mileage_per_year': 0,
            'engine_size_cc_log': 0,
            'horse_power_log': 0,
            'torque_log': 0,
            'power_per_cc': 0,
            'torque_per_cc': 0,
            'power_to_weight': 0,
            'power_to_torque': 0,
            'performance_score': 0
        }
        
        # Handle NaN values in numeric columns
        for col in self.numeric_features + self.derived_features + ['annual_insurance']:
            if col in df.columns:
                if df[col].isna().any():
                    print(f"Warning: Found NaN in {col} during prefix addition, filling with {defaults.get(col, 0)}")
                    df[col] = df[col].fillna(defaults.get(col, 0))
        
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
            
        # Final NaN check and fill
        for col in result.columns:
            if col.startswith('num_'):
                base_col = col.split('__')[-1]
                if result[col].isna().any():
                    print(f"Warning: Filling NaN in {col} with default value {defaults.get(base_col, 0)}")
                    result[col] = result[col].fillna(defaults.get(base_col, 0))
        
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
        
    # Analyze model features first
    print("\nAnalyzing model features...")
    feature_categories = inspect_model_features()
    
    # Convert dictionary to DataFrame if needed
    if isinstance(input_data, dict):
        # Map input features to expected ranges
        mapped_data = map_input_features(input_data)
        input_data = pd.DataFrame([mapped_data])
    
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

def inspect_model_features():
    """Analyze and print information about the model's features"""
    global _current_model
    
    if _current_model is None:
        load_model()
    
    if not hasattr(_current_model, 'feature_names_in_'):
        print("Model does not expose feature names")
        return
    
    feature_names = _current_model.feature_names_in_
    
    # Categorize features by prefix
    feature_categories = {
        'num_insurance': [],
        'num_main': [],
        'cat': []
    }
    
    for feature in feature_names:
        if feature.startswith('num_insurance__'):
            feature_categories['num_insurance'].append(feature)
        elif feature.startswith('num_main__'):
            feature_categories['num_main'].append(feature)
        elif feature.startswith('cat__'):
            feature_categories['cat'].append(feature)
    
    print("\nModel Feature Analysis:")
    print("======================")
    
    print("\nNumeric Insurance Features:")
    for f in feature_categories['num_insurance']:
        print(f"- {f}")
    
    print("\nMain Numeric Features:")
    print("Basic Features:")
    basic = [f for f in feature_categories['num_main'] 
             if not any(x in f for x in ['squared', 'log', 'per', 'power', 'score', 'factor'])]
    for f in basic:
        print(f"- {f}")
    
    print("\nDerived Features:")
    derived = [f for f in feature_categories['num_main'] 
              if any(x in f for x in ['squared', 'log', 'per', 'power', 'score', 'factor'])]
    for f in derived:
        print(f"- {f}")
    
    print("\nCategorical Features:")
    for f in feature_categories['cat']:
        print(f"- {f}")
    
    return feature_categories

def map_input_features(input_dict: dict) -> dict:
    """Map raw input features to expected model features"""
    features = ModelFeatures()
    
    # Standard feature ranges
    feature_ranges = {
        'car_age': (0, 50),
        'mileage_num': (0, 1000000),
        'engine_size_cc_num': (600, 8000),
        'horse_power_num': (30, 1000),
        'torque_num': (30, 1200),
        'seats_num': (2, 15),
        'annual_insurance': (10000, 500000)
    }
    
    # Default values for categorical features
    categorical_defaults = {
        'make_name_cleaned': 'toyota',
        'model_name_cleaned': 'other',
        'fuel_type_cleaned': 'petrol',
        'transmission_cleaned': 'manual',
        'drive_type_cleaned': '2wd',
        'body_type_cleaned': 'sedan',
        'usage_type_clean': 'Foreign Used'
    }
    
    # Clip values to valid ranges
    mapped = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in input_dict:
            mapped[feature] = np.clip(float(input_dict[feature]), min_val, max_val)
    
    # Map categorical features
    for feature, default_value in categorical_defaults.items():
        if feature in input_dict:
            value = str(input_dict[feature]).lower()
            if feature in features.known_categories:
                if value in features.known_categories[feature]:
                    mapped[feature] = value
                else:
                    if feature == 'make_name_cleaned':
                        print(f"Warning: Unknown make '{value}', using 'other'")
                        mapped[feature] = 'other'
                    else:
                        print(f"Warning: Unknown {feature} value '{value}', using default '{default_value}'")
                        mapped[feature] = default_value
            else:
                mapped[feature] = value
        else:
            mapped[feature] = default_value
    
    # Print feature mapping debug info
    print("\nFeature Mapping Analysis:")
    print("========================")
    print("\nNumeric Features:")
    for feature in feature_ranges.keys():
        if feature in input_dict:
            original = input_dict[feature]
            mapped_value = mapped[feature]
            if original != mapped_value:
                print(f"- {feature}: Original={original}, Mapped={mapped_value} (clipped to valid range)")
            else:
                print(f"- {feature}: {mapped_value} (within valid range)")
        else:
            print(f"- {feature}: Missing - using default value")
    
    print("\nCategorical Features:")
    for feature in categorical_defaults.keys():
        if feature in input_dict:
            original = input_dict[feature]
            mapped_value = mapped[feature]
            if original.lower() != mapped_value:
                print(f"- {feature}: Original='{original}', Mapped='{mapped_value}' (standardized)")
            else:
                print(f"- {feature}: '{mapped_value}' (valid category)")
        else:
            print(f"- {feature}: Missing - using default '{categorical_defaults[feature]}'")
    
    return mapped
