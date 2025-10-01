import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_models()

# Title and description
st.title("🏠 House Price Estimator")
st.markdown("### Predict house prices based on property features")
st.divider()

if model is not None and scaler is not None:
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Property Details")
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.5, step=0.5)
        sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
        sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=50000, value=5000)
        floors = st.number_input("Floors", min_value=1.0, max_value=3.5, value=1.0, step=0.5)
        sqft_above = st.number_input("Above Ground (sqft)", min_value=500, max_value=10000, value=1800)
        sqft_basement = st.number_input("Basement (sqft)", min_value=0, max_value=5000, value=200)
        
    with col2:
        st.subheader("🌟 Property Features")
        waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        view = st.slider("View Rating", 0, 4, 0)
        condition = st.slider("Condition", 1, 5, 3)
        grade = st.slider("Grade", 1, 13, 7)
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1990)
        yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📍 Location")
        zipcode = st.number_input("Zipcode", min_value=98001, max_value=98199, value=98103)
        lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.65, format="%.6f")
        long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.35, format="%.6f")
        
    with col4:
        st.subheader("🏘️ Neighborhood & Sale Info")
        sqft_living15 = st.number_input("Nearby Living Area (sqft)", min_value=500, max_value=10000, value=2200)
        sqft_lot15 = st.number_input("Nearby Lot Size (sqft)", min_value=500, max_value=50000, value=4800)
        sale_year = st.number_input("Sale Year", min_value=2000, max_value=2030, value=2023)
        sale_month = st.selectbox("Sale Month", list(range(1, 13)), index=9)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
        # Prepare data
        data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'grade': grade,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'zipcode': zipcode,
            'lat': lat,
            'long': long,
            'sqft_living15': sqft_living15,
            'sqft_lot15': sqft_lot15,
            'sale_year': sale_year,
            'sale_month': sale_month
        }
        
        try:
            # Feature engineering
            input_df = pd.DataFrame([data])
            input_df['age'] = input_df['sale_year'] - input_df['yr_built']
            input_df['renovated'] = input_df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
            
            # Feature selection
            features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                       'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                       'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
                       'age', 'renovated', 'sale_year', 'sale_month']
            
            input_df = input_df[features]
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.success("### 🎉 Prediction Complete!")
            st.metric(
                label="Estimated House Price",
                value=f"${prediction:,.2f}",
                delta=None
            )
            
            # Additional insights
            st.info(f"""
            **Property Summary:**
            - {bedrooms} bed, {bathrooms} bath
            - {sqft_living:,} sqft living space
            - Built in {yr_built} (Age: {sale_year - yr_built} years)
            - Grade: {grade}/13, Condition: {condition}/5
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.error("⚠️ Could not load model files. Please ensure `model.pkl` and `scaler.pkl` are in the same directory.")

