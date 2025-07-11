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
        # Feature definitions with validation rules and defaults
        self.feature_specs = {
            # Basic numeric features
            'annual_insurance': {'type': 'numeric', 'prefix': 'num_insurance', 'min': 10000, 'max': 500000, 'default': 40000},
            'car_age': {'type': 'numeric', 'prefix': 'num_main', 'min': 0, 'max': 50, 'default': 5},
            'mileage_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 0, 'max': 1000000, 'default': 50000},
            'engine_size_cc_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 600, 'max': 8000, 'default': 1500},
            'horse_power_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 30, 'max': 1000, 'default': 100},
            'torque_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 30, 'max': 1200, 'default': 150},
            'acceleration_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 0, 'max': 30, 'default': 12.0},
            'seats_num': {'type': 'numeric', 'prefix': 'num_main', 'min': 2, 'max': 15, 'default': 5},
            'is_luxury_make': {'type': 'numeric', 'prefix': 'num_main', 'min': 0, 'max': 1, 'default': 0},
            
            # Categorical features with valid values
            'make_name_cleaned': {
                'type': 'categorical', 
                'prefix': 'cat',
                'valid_values': [
                    'toyota', 'honda', 'mazda', 'nissan', 'suzuki', 'mitsubishi', 'subaru',  # Japanese
                    'bmw', 'mercedes', 'audi', 'volkswagen', 'porsche',  # German
                    'lexus', 'land rover', 'jaguar',  # Luxury
                    'hyundai', 'kia',  # Korean
                    'other'  # Fallback
                ],
                'default': 'toyota'
            },
            'fuel_type_cleaned': {
                'type': 'categorical',
                'prefix': 'cat',
                'valid_values': ['petrol', 'diesel', 'hybrid_petrol', 'hybrid_diesel', 'electric', 'plugin_hybrid_petrol', 'unknown'],
                'default': 'petrol'
            },
            'transmission_cleaned': {
                'type': 'categorical',
                'prefix': 'cat',
                'valid_values': ['automatic', 'manual', 'automated_manual', 'unknown'],
                'default': 'manual'
            },
            'drive_type_cleaned': {
                'type': 'categorical',
                'prefix': 'cat',
                'valid_values': ['2wd', '2wd_front', '2wd_mid_engine', '2wd_rear', '4wd', 'awd', 'unknown'],
                'default': '2wd'
            },
            'body_type_cleaned': {
                'type': 'categorical',
                'prefix': 'cat',
                'valid_values': ['sedan', 'suv', 'hatchback', 'wagon', 'van_minivan', 'pickup_truck', 'coupe', 'bus', 'convertible', 'other'],
                'default': 'sedan'
            },
            'usage_type_clean': {
                'type': 'categorical',
                'prefix': 'cat',
                'valid_values': ['Foreign Used', 'Kenyan Used'],
                'default': 'Foreign Used'
            }
        }
        
        # Derived feature definitions
        self.derived_features = {
            'car_age_squared': {'type': 'derived', 'prefix': 'num_main', 'base': 'car_age', 'operation': 'square'},
            'mileage_log': {'type': 'derived', 'prefix': 'num_main', 'base': 'mileage_num', 'operation': 'log'},
            'mileage_per_year': {'type': 'derived', 'prefix': 'num_main', 'base': ['mileage_num', 'car_age'], 'operation': 'divide'},
            'engine_size_cc_log': {'type': 'derived', 'prefix': 'num_main', 'base': 'engine_size_cc_num', 'operation': 'log'},
            'horse_power_log': {'type': 'derived', 'prefix': 'num_main', 'base': 'horse_power_num', 'operation': 'log'},
            'torque_log': {'type': 'derived', 'prefix': 'num_main', 'base': 'torque_num', 'operation': 'log'},
            'power_per_cc': {'type': 'derived', 'prefix': 'num_main', 'base': ['horse_power_num', 'engine_size_cc_num'], 'operation': 'divide'},
            'mileage_per_cc': {'type': 'derived', 'prefix': 'num_main', 'base': ['mileage_num', 'engine_size_cc_num'], 'operation': 'divide'},
            'power_to_weight': {'type': 'derived', 'prefix': 'num_main', 'base': 'horse_power_num', 'operation': 'power_to_weight'},
            'power_to_torque': {'type': 'derived', 'prefix': 'num_main', 'base': ['horse_power_num', 'torque_num'], 'operation': 'divide'},
            'performance_score': {'type': 'derived', 'prefix': 'num_main', 'base': ['power_to_weight', 'power_to_torque'], 'operation': 'performance'}
        }
        
        # Market segment definitions
        self.market_segments = {
            'luxury': ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar'],
            'premium': ['toyota', 'honda', 'volkswagen', 'mazda', 'subaru'],
            'economy': ['suzuki', 'mitsubishi', 'nissan', 'hyundai', 'kia']
        }
        
        # Initialize feature lists from specs
        self.numeric_features = [name for name, spec in self.feature_specs.items() if spec['type'] == 'numeric']
        self.categorical_features = [name for name, spec in self.feature_specs.items() if spec['type'] == 'categorical']

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

    def preprocess_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Comprehensive feature preprocessing pipeline
        """
        df = data.copy()
        warnings = []
        adjustment_factors = {}
        
        try:
            # 1. Validate and process basic features
            df = self._process_basic_features(df, warnings)
            
            # 2. Calculate derived features
            df = self._calculate_derived_features(df, warnings)
            
            # 3. Calculate market factors
            adjustment_factors = self._calculate_market_factors(df)
            
            # 4. Add prefixes and final validation
            df = self._add_feature_prefixes(df)
            
            # 5. Final NaN check
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                raise ValueError(f"NaN values found in columns after preprocessing: {nan_cols}")
            
            return df, adjustment_factors, warnings
            
        except Exception as e:
            raise ValueError(f"Feature preprocessing failed: {str(e)}")

    def _process_basic_features(self, df: pd.DataFrame, warnings: list) -> pd.DataFrame:
        """Process and validate basic features"""
        for feature, spec in self.feature_specs.items():
            # Ensure feature exists
            if feature not in df.columns:
                print(f"Adding missing feature {feature} with default value {spec['default']}")
                df[feature] = spec['default']
            
            if spec['type'] == 'numeric':
                # Convert to numeric and handle invalid values
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(spec['default'])
                
                # Clip to valid range
                original_value = df[feature].iloc[0]
                df[feature] = df[feature].clip(lower=spec['min'], upper=spec['max'])
                if df[feature].iloc[0] != original_value:
                    warnings.append(f"{feature} value {original_value} was clipped to {df[feature].iloc[0]}")
                
            elif spec['type'] == 'categorical':
                # Convert to lowercase and validate
                df[feature] = df[feature].astype(str).str.lower()
                if df[feature].iloc[0] not in spec['valid_values']:
                    warnings.append(f"Invalid {feature} value: {df[feature].iloc[0]}, using default: {spec['default']}")
                    df[feature] = spec['default']
        
        return df

    def _calculate_derived_features(self, df: pd.DataFrame, warnings: list) -> pd.DataFrame:
        """Calculate all derived features"""
        for feature, spec in self.derived_features.items():
            try:
                if spec['operation'] == 'square':
                    df[feature] = df[spec['base']].clip(lower=0) ** 2
                elif spec['operation'] == 'log':
                    df[feature] = np.log1p(df[spec['base']].clip(lower=0))
                elif spec['operation'] == 'divide':
                    numerator, denominator = spec['base']
                    df[feature] = (df[numerator] / df[denominator].clip(lower=1)).clip(lower=0)
                elif spec['operation'] == 'power_to_weight':
                    # Estimate weight based on engine size and vehicle type
                    base_weight = df['engine_size_cc_num'] * 0.9
                    body_type_factors = {'suv': 1.2, 'pickup_truck': 1.3, 'bus': 2.0, 'van_minivan': 1.4}
                    weight_factor = body_type_factors.get(df['body_type_cleaned'].iloc[0], 1.0)
                    estimated_weight = base_weight * weight_factor
                    df[feature] = (df[spec['base']] / estimated_weight.clip(lower=1)).clip(lower=0, upper=1)
                elif spec['operation'] == 'performance':
                    df[feature] = (df[spec['base'][0]] * df[spec['base'][1]]).clip(lower=0) ** 0.5
            except Exception as e:
                warnings.append(f"Error calculating {feature}: {str(e)}")
                df[feature] = 0
        
        return df

    def _calculate_market_factors(self, df: pd.DataFrame) -> dict:
        """Calculate market adjustment factors"""
        factors = {}
        
        # Brand factor based on market segment
        make = df['make_name_cleaned'].iloc[0]
        if make in self.market_segments['luxury']:
            factors['brand_factor'] = 1.3
            df['is_luxury_make'] = 1
        elif make in self.market_segments['premium']:
            factors['brand_factor'] = 1.1
            df['is_luxury_make'] = 0
        else:
            factors['brand_factor'] = 0.9
            df['is_luxury_make'] = 0
        
        # Age and mileage factors
        factors['age_factor'] = 0.90 ** df['car_age'].iloc[0]
        factors['mileage_factor'] = 0.95 ** (df['mileage_num'].iloc[0] / 10000)
        
        # Usage factor
        usage = df['usage_type_clean'].iloc[0]
        factors['usage_factor'] = 1.1 if usage == 'Foreign Used' else 0.9 if usage == 'Kenyan Used' else 1.0
        
        # Calculate total depreciation
        factors['total_depreciation'] = factors['age_factor'] * factors['mileage_factor'] * factors['brand_factor']
        
        return factors

    def _add_feature_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add proper prefixes to all features"""
        renamed = {}
        
        # Add prefixes based on feature specifications
        for feature, spec in self.feature_specs.items():
            if feature in df.columns:
                renamed[feature] = f"{spec['prefix']}__{feature}"
        
        # Add prefixes for derived features
        for feature, spec in self.derived_features.items():
            if feature in df.columns:
                renamed[feature] = f"{spec['prefix']}__{feature}"
        
        return df.rename(columns=renamed)
class FeatureMapper:
    """Maps user inputs to model features"""
    
    def __init__(self):
        # Define mappings between user inputs and model features
        self.input_to_model_mapping = {
            # Direct numeric mappings
            'insurance': {'model_feature': 'num_insurance__annual_insurance', 'type': 'numeric'},
            'car_age': {'model_feature': 'num_main__car_age', 'type': 'numeric'},
            'mileage': {'model_feature': 'num_main__mileage_num', 'type': 'numeric'},
            'engine_size': {'model_feature': 'num_main__engine_size_cc_num', 'type': 'numeric'},
            'horsepower': {'model_feature': 'num_main__horse_power_num', 'type': 'numeric'},
            'torque': {'model_feature': 'num_main__torque_num', 'type': 'numeric'},
            'seats': {'model_feature': 'num_main__seats_num', 'type': 'numeric'},
            'acceleration': {'model_feature': 'num_main__acceleration_num', 'type': 'numeric'},
            
            # Categorical mappings
            'make': {'model_feature': 'cat__make_name_cleaned', 'type': 'categorical'},
            'model': {'model_feature': 'cat__model_name_cleaned', 'type': 'categorical'},
            'body_type': {'model_feature': 'cat__body_type_cleaned', 'type': 'categorical'},
            'fuel_type': {'model_feature': 'cat__fuel_type_cleaned', 'type': 'categorical'},
            'transmission': {'model_feature': 'cat__transmission_cleaned', 'type': 'categorical'},
            'drive_type': {'model_feature': 'cat__drive_type_cleaned', 'type': 'categorical'},
            'usage_type': {'model_feature': 'cat__usage_type_clean', 'type': 'categorical'}
        }
        
        # Define derived feature calculations
        self.derived_features = {
            'num_main__car_age_squared': lambda x: x['car_age'] ** 2,
            'num_main__mileage_log': lambda x: np.log1p(x['mileage']),
            'num_main__mileage_per_year': lambda x: x['mileage'] / max(x['car_age'], 1),
            'num_main__engine_size_cc_log': lambda x: np.log1p(x['engine_size']),
            'num_main__horse_power_log': lambda x: np.log1p(x['horsepower']),
            'num_main__torque_log': lambda x: np.log1p(x['torque']),
            'num_main__power_per_cc': lambda x: x['horsepower'] / max(x['engine_size'], 1),
            'num_main__mileage_per_cc': lambda x: x['mileage'] / max(x['engine_size'], 1),
            'num_main__is_luxury_make': lambda x: 1 if x['make'].lower() in ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover', 'jaguar'] else 0
        }
        
        # Define value ranges and defaults
        self.feature_ranges = {
            'insurance': {'min': 10000, 'max': 500000, 'default': 40000},
            'car_age': {'min': 0, 'max': 50, 'default': 5},
            'mileage': {'min': 0, 'max': 1000000, 'default': 50000},
            'engine_size': {'min': 600, 'max': 8000, 'default': 1500},
            'horsepower': {'min': 30, 'max': 1000, 'default': 100},
            'torque': {'min': 30, 'max': 1200, 'default': 150},
            'acceleration': {'min': 0, 'max': 30, 'default': 12.0},
            'seats': {'min': 2, 'max': 15, 'default': 5}
        }
        
        # Define categorical value mappings
        self.categorical_mappings = {
            'make': {
                'valid_values': ['toyota', 'honda', 'mazda', 'nissan', 'suzuki', 'mitsubishi', 'subaru',
                               'bmw', 'mercedes', 'audi', 'volkswagen', 'porsche', 'lexus', 'land rover',
                               'jaguar', 'hyundai', 'kia'],
                'default': 'toyota',
                'other_value': 'other'
            },
            'fuel_type': {
                'valid_values': ['petrol', 'diesel', 'hybrid_petrol', 'hybrid_diesel', 'electric'],
                'default': 'petrol',
                'other_value': 'unknown'
            },
            'transmission': {
                'valid_values': ['automatic', 'manual', 'automated_manual'],
                'default': 'manual',
                'other_value': 'unknown'
            },
            'drive_type': {
                'valid_values': ['2wd', '4wd', 'awd'],
                'default': '2wd',
                'other_value': 'unknown'
            },
            'body_type': {
                'valid_values': ['sedan', 'suv', 'hatchback', 'wagon', 'van_minivan', 'pickup_truck', 
                               'coupe', 'bus', 'convertible'],
                'default': 'sedan',
                'other_value': 'other'
            },
            'usage_type': {
                'valid_values': ['Foreign Used', 'Kenyan Used'],
                'default': 'Foreign Used',
                'other_value': 'Foreign Used'
            }
        }

    def map_input_to_features(self, user_input: dict) -> pd.DataFrame:
        """Maps user input to model features"""
        # Initialize storage for mapped features
        mapped_features = {}
        warnings = []
        
        # Process direct mappings
        for input_name, mapping in self.input_to_model_mapping.items():
            if mapping['type'] == 'numeric':
                # Handle numeric features
                value = user_input.get(input_name)
                if value is None:
                    value = self.feature_ranges[input_name]['default']
                    warnings.append(f"Using default value {value} for {input_name}")
                else:
                    value = float(value)
                    # Clip to valid range
                    min_val = self.feature_ranges[input_name]['min']
                    max_val = self.feature_ranges[input_name]['max']
                    if value < min_val or value > max_val:
                        old_value = value
                        value = np.clip(value, min_val, max_val)
                        warnings.append(f"Clipped {input_name} from {old_value} to {value}")
                
                mapped_features[mapping['model_feature']] = value
                
            elif mapping['type'] == 'categorical':
                # Handle categorical features
                value = str(user_input.get(input_name, '')).lower()
                cat_info = self.categorical_mappings.get(input_name, {})
                
                if value not in cat_info.get('valid_values', []):
                    old_value = value
                    value = cat_info.get('other_value', cat_info.get('default', value))
                    warnings.append(f"Mapped invalid {input_name} '{old_value}' to '{value}'")
                
                mapped_features[mapping['model_feature']] = value
        
        # Calculate derived features
        temp_input = {k: user_input.get(k, self.feature_ranges.get(k, {}).get('default', 0)) 
                     for k in ['car_age', 'mileage', 'engine_size', 'horsepower', 'torque', 'make']}
        
        for feature_name, calculation in self.derived_features.items():
            try:
                mapped_features[feature_name] = calculation(temp_input)
            except Exception as e:
                warnings.append(f"Error calculating {feature_name}: {str(e)}")
                mapped_features[feature_name] = 0
        
        return pd.DataFrame([mapped_features]), warnings

    def get_required_inputs(self) -> dict:
        """Returns the list of required inputs with their specifications"""
        return {
            'numeric_inputs': self.feature_ranges,
            'categorical_inputs': self.categorical_mappings
        }

# Update ModelFeatures to use FeatureMapper
def predict_price(input_data: Union[Dict, pd.DataFrame]) -> tuple[float, List[str]]:
    """Make a prediction using the current model with input mapping"""
    global _current_model
    
    # Load model if not already loaded
    if _current_model is None:
        load_model()
    
    try:
        # Map input to features
        mapper = FeatureMapper()
        if isinstance(input_data, dict):
            mapped_data, warnings = mapper.map_input_to_features(input_data)
        else:
            # If already a DataFrame, assume it's in the correct format
            mapped_data = input_data
            warnings = []
        
        # Create categorical features
        all_features = pd.DataFrame()
        
        # Add one-hot encoded features
        for col in mapped_data.columns:
            if col.startswith('cat__'):
                feature = col.split('__')[1]
                value = mapped_data[col].iloc[0]
                cat_info = mapper.categorical_mappings.get(feature.split('_')[0], {})
                valid_values = cat_info.get('valid_values', [value])
                
                for val in valid_values:
                    col_name = f"{col}_{val}"
                    all_features[col_name] = 1 if value == val else 0
            else:
                all_features[col] = mapped_data[col]
        
        # Ensure all model features are present
        if hasattr(_current_model, 'feature_names_in_'):
            required_features = set(_current_model.feature_names_in_)
            current_features = set(all_features.columns)
            
            # Add missing features with zeros
            missing_features = required_features - current_features
            if missing_features:
                for feature in missing_features:
                    all_features[feature] = 0
            
            # Ensure correct column order
            all_features = all_features[_current_model.feature_names_in_]
        
        # Make prediction
        prediction = _current_model.predict(all_features)
        
        # Convert prediction if needed
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction = np.exp(prediction[0])
        
        return prediction, warnings
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")
