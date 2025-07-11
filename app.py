import streamlit as st
import pandas as pd
import numpy as np
from model_utils import predict_price, load_model
from feature_impact import analyze_feature_sensitivity, get_feature_impact
import plotly.graph_objects as go

st.set_page_config(page_title="AUTO APPRAISAL", layout="wide")

def main():
    st.title("🚗 CAR PRICE PREDICTION")
    
    # Load model in background
    with st.spinner("Loading model..."):
        try:
            load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

    # --- Main Area for User Input ---
    st.header("📝 ENTER CAR FEATURES")

    # Input form with all required fields, organized in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Vehicle Identity")
        make = st.text_input("Make", "toyota", help="e.g., toyota, honda, bmw").lower()
        model = st.text_input("Model", "axio", help="e.g., axio, fit, 3 series").lower()
        car_age = st.number_input("Car Age (Years)", min_value=0, max_value=50, value=5, step=1)
        usage_type = st.selectbox(
            label="Usage Type",
            options=["Foreign Used", "Kenyan Used"],
            help="Condition based on import status."
        )
        
    with col2:
        st.subheader("Performance Specs")
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
        engine_size = st.number_input("Engine Size (cc)", min_value=500, max_value=8000, value=1500, step=50)
        horse_power = st.number_input("Horse Power (HP)", min_value=30, max_value=1000, value=110, step=5)
        torque = st.number_input("Torque (Nm)", min_value=30, max_value=1200, value=140, step=5)

    with col3:
        st.subheader("Configuration")
        body_type = st.selectbox(
            label="Body Type",
            options=["sedan", "suv", "hatchback", "wagon", "van_minivan", "pickup_truck", "coupe", "bus", "convertible", "other"]
        )
        fuel_type = st.selectbox(
            label="Fuel Type",
            options=["petrol", "diesel", "hybrid_petrol", "hybrid_diesel", "electric"]
        )
        transmission = st.selectbox(
            label="Transmission",
            options=["automatic", "manual", "automated_manual"]
        )
        drive_type = st.selectbox(
            label="Drive Type",
            options=["2wd", "4wd", "awd"]
        )

    st.divider()
    
    # --- Optional and Financial Inputs ---
    st.subheader("Optional & Financial Details")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        seats = st.number_input("Number of Seats", min_value=2, max_value=15, value=5, step=1)
    with col_opt2:
        insurance = st.number_input("Estimated Annual Insurance (KES)", min_value=0, value=40000, step=1000)

    # Store previous inputs in session state
    if 'previous_input' not in st.session_state:
        st.session_state['previous_input'] = None
        
    # --- Prediction Trigger and Display ---
    if st.button("🔍 Predict Price", type="primary", use_container_width=True):
        
        # Create the raw input dictionary for the prediction pipeline
        input_dict = {
            'make_name_cleaned': make,
            'model_name_cleaned': model,
            'car_age': float(car_age),
            'mileage_num': float(mileage),
            'engine_size_cc_num': float(engine_size),
            'horse_power_num': float(horse_power),
            'torque_num': float(torque),
            'body_type_cleaned': body_type,
            'fuel_type_cleaned': fuel_type,
            'transmission_cleaned': transmission,
            'drive_type_cleaned': drive_type,
            'usage_type_clean': usage_type,
            'seats_num': float(seats),
            'annual_insurance': float(insurance),
            'condition_clean': 'used'  # Default value
        }

        try:
            with st.spinner("⌛ Analyzing features and predicting price..."):
                predicted_price = predict_price(input_dict)
            
            st.success("✅ Prediction Successful!")
            
            # Display prediction with formatting
            st.metric("💰 Predicted Market Price", f"KES {predicted_price:,.0f}")
            
            # Compare with previous prediction if available
            if st.session_state['previous_input'] is not None:
                impact, old_price, new_price = get_feature_impact(
                    st.session_state['previous_input'], 
                    input_dict,
                    predict_price
                )
                
                if impact:
                    st.subheader("📊 What Changed?")
                    price_change = new_price - old_price
                    percent_change = (price_change / old_price) * 100
                    
                    # Show price change with color
                    if abs(percent_change) >= 0.1:
                        direction = "+" if price_change > 0 else ""
                        st.metric(
                            "Price Change",
                            f"KES {direction}{price_change:,.0f}",
                            f"{direction}{percent_change:.1f}%"
                        )
                        
                        # Explain the impact
                        st.markdown("### Impact Analysis")
                        from feature_impact import format_impact
                        st.markdown(format_impact(impact))
            
            # Analyze sensitivity for numeric features
            st.subheader("📈 FEATURE SENSITIVITY ANALYSIS")
            numeric_features = ['car_age', 'mileage_num', 'engine_size_cc_num', 'horse_power_num']
            
            for feature in numeric_features:
                values, predictions = analyze_feature_sensitivity(input_dict, feature, predict_price)
                if any(p is not None for p in predictions):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=values,
                        y=predictions,
                        mode='lines+markers',
                        name=f'Price vs {feature}'
                    ))
                    fig.update_layout(
                        title=f'How {feature} Affects Price',
                        xaxis_title=feature,
                        yaxis_title='Predicted Price (KES)',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Store current input for next comparison
            st.session_state['previous_input'] = input_dict
            
            # Show a confidence range for context
            st.info(f"""
            **📋 Estimated Price Range:**
            - 🔻 Conservative: KES {predicted_price * 0.9:,.0f}
            - ⭐ Most Likely: KES {predicted_price:,.0f}
            - 🔺 Optimistic: KES {predicted_price * 1.1:,.0f}
            
            *This range is an estimate and not a guarantee.*
            """)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.error("Please ensure all input values are reasonable and try again.")

if __name__ == "__main__":
    main()
