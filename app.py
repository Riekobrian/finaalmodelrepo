import streamlit as st
import pandas as pd
from model_utils import predict_price, FeatureMapper

def main():
    st.title("Car Price Prediction")
    
    # Initialize feature mapper
    mapper = FeatureMapper()
    input_specs = mapper.get_required_inputs()
    
    # Create form for user input
    with st.form("prediction_form"):
        # Basic car information
        col1, col2 = st.columns(2)
        
        with col1:
            make = st.selectbox(
                "Make",
                options=input_specs['categorical_inputs']['make']['valid_values'],
                index=input_specs['categorical_inputs']['make']['valid_values'].index(
                    input_specs['categorical_inputs']['make']['default']
                )
            )
            
            body_type = st.selectbox(
                "Body Type",
                options=input_specs['categorical_inputs']['body_type']['valid_values'],
                index=input_specs['categorical_inputs']['body_type']['valid_values'].index(
                    input_specs['categorical_inputs']['body_type']['default']
                )
            )
            
            transmission = st.selectbox(
                "Transmission",
                options=input_specs['categorical_inputs']['transmission']['valid_values'],
                index=input_specs['categorical_inputs']['transmission']['valid_values'].index(
                    input_specs['categorical_inputs']['transmission']['default']
                )
            )
            
            drive_type = st.selectbox(
                "Drive Type",
                options=input_specs['categorical_inputs']['drive_type']['valid_values'],
                index=input_specs['categorical_inputs']['drive_type']['valid_values'].index(
                    input_specs['categorical_inputs']['drive_type']['default']
                )
            )
        
        with col2:
            fuel_type = st.selectbox(
                "Fuel Type",
                options=input_specs['categorical_inputs']['fuel_type']['valid_values'],
                index=input_specs['categorical_inputs']['fuel_type']['valid_values'].index(
                    input_specs['categorical_inputs']['fuel_type']['default']
                )
            )
            
            usage_type = st.selectbox(
                "Usage Type",
                options=input_specs['categorical_inputs']['usage_type']['valid_values'],
                index=input_specs['categorical_inputs']['usage_type']['valid_values'].index(
                    input_specs['categorical_inputs']['usage_type']['default']
                )
            )
            
            car_age = st.number_input(
                "Car Age (years)",
                min_value=input_specs['numeric_inputs']['car_age']['min'],
                max_value=input_specs['numeric_inputs']['car_age']['max'],
                value=input_specs['numeric_inputs']['car_age']['default']
            )
            
            mileage = st.number_input(
                "Mileage (km)",
                min_value=input_specs['numeric_inputs']['mileage']['min'],
                max_value=input_specs['numeric_inputs']['mileage']['max'],
                value=input_specs['numeric_inputs']['mileage']['default']
            )
        
        # Technical specifications
        st.subheader("Technical Specifications")
        col3, col4 = st.columns(2)
        
        with col3:
            engine_size = st.number_input(
                "Engine Size (cc)",
                min_value=input_specs['numeric_inputs']['engine_size']['min'],
                max_value=input_specs['numeric_inputs']['engine_size']['max'],
                value=input_specs['numeric_inputs']['engine_size']['default']
            )
            
            horsepower = st.number_input(
                "Horsepower",
                min_value=input_specs['numeric_inputs']['horsepower']['min'],
                max_value=input_specs['numeric_inputs']['horsepower']['max'],
                value=input_specs['numeric_inputs']['horsepower']['default']
            )
        
        with col4:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=input_specs['numeric_inputs']['torque']['min'],
                max_value=input_specs['numeric_inputs']['torque']['max'],
                value=input_specs['numeric_inputs']['torque']['default']
            )
            
            acceleration = st.number_input(
                "0-100 km/h (seconds)",
                min_value=input_specs['numeric_inputs']['acceleration']['min'],
                max_value=input_specs['numeric_inputs']['acceleration']['max'],
                value=input_specs['numeric_inputs']['acceleration']['default']
            )
        
        # Other specifications
        st.subheader("Other Specifications")
        col5, col6 = st.columns(2)
        
        with col5:
            seats = st.number_input(
                "Number of Seats",
                min_value=input_specs['numeric_inputs']['seats']['min'],
                max_value=input_specs['numeric_inputs']['seats']['max'],
                value=input_specs['numeric_inputs']['seats']['default']
            )
        
        with col6:
            insurance = st.number_input(
                "Annual Insurance (KES)",
                min_value=input_specs['numeric_inputs']['insurance']['min'],
                max_value=input_specs['numeric_inputs']['insurance']['max'],
                value=input_specs['numeric_inputs']['insurance']['default']
            )
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Prepare input data
        input_data = {
            'make': make,
            'body_type': body_type,
            'transmission': transmission,
            'drive_type': drive_type,
            'fuel_type': fuel_type,
            'usage_type': usage_type,
            'car_age': car_age,
            'mileage': mileage,
            'engine_size': engine_size,
            'horsepower': horsepower,
            'torque': torque,
            'acceleration': acceleration,
            'seats': seats,
            'insurance': insurance
        }
        
        try:
            # Get prediction and warnings
            prediction, warnings = predict_price(input_data)
            
            # Display prediction
            st.success(f"Estimated Price: KES {prediction:,.2f}")
            
            # Display warnings if any
            if warnings:
                st.warning("Note:")
                for warning in warnings:
                    st.write(f"- {warning}")
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    main()
