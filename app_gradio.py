import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    model, scaler = None, None

def predict_price(bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
                 view, condition, grade, sqft_above, sqft_basement, yr_built, 
                 yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15, 
                 sale_year, sale_month):
    """Predict house price based on input features"""
    
    if model is None or scaler is None:
        return "❌ Error: Model not loaded properly"
    
    try:
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
        
        # Format output
        result = f"""
🎉 **Prediction Complete!**

💰 **Estimated House Price: ${prediction:,.2f}**

📊 **Property Summary:**
- {bedrooms} bed, {bathrooms} bath
- {sqft_living:,} sqft living space
- Built in {yr_built} (Age: {sale_year - yr_built} years)
- Grade: {grade}/13, Condition: {condition}/5
- {'Waterfront' if waterfront == 1 else 'No Waterfront'}
"""
        return result
        
    except Exception as e:
        return f"❌ Prediction error: {e}"

# Create Gradio interface
with gr.Blocks(title="House Price Estimator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 House Price Estimator")
    gr.Markdown("### Predict house prices based on property features")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 📐 Property Details")
            bedrooms = gr.Number(label="Bedrooms", value=3)
            bathrooms = gr.Number(label="Bathrooms", value=2.5)
            sqft_living = gr.Number(label="Living Area (sqft)", value=2000)
            sqft_lot = gr.Number(label="Lot Size (sqft)", value=5000)
            floors = gr.Number(label="Floors", value=1)
            sqft_above = gr.Number(label="Above Ground (sqft)", value=1800)
            sqft_basement = gr.Number(label="Basement (sqft)", value=200)
        
        with gr.Column():
            gr.Markdown("#### 🌟 Property Features")
            waterfront = gr.Radio([0, 1], label="Waterfront (0=No, 1=Yes)", value=0)
            view = gr.Slider(0, 4, step=1, label="View Rating", value=0)
            condition = gr.Slider(1, 5, step=1, label="Condition", value=3)
            grade = gr.Slider(1, 13, step=1, label="Grade", value=7)
            yr_built = gr.Number(label="Year Built", value=1990)
            yr_renovated = gr.Number(label="Year Renovated (0 if never)", value=0)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 📍 Location")
            zipcode = gr.Number(label="Zipcode", value=98103)
            lat = gr.Number(label="Latitude", value=47.65)
            long = gr.Number(label="Longitude", value=-122.35)
        
        with gr.Column():
            gr.Markdown("#### 🏘️ Neighborhood & Sale Info")
            sqft_living15 = gr.Number(label="Nearby Living Area (sqft)", value=2200)
            sqft_lot15 = gr.Number(label="Nearby Lot Size (sqft)", value=4800)
            sale_year = gr.Number(label="Sale Year", value=2023)
            sale_month = gr.Slider(1, 12, step=1, label="Sale Month", value=10)
    
    predict_btn = gr.Button("🔮 Predict House Price", variant="primary", size="lg")
    output = gr.Markdown(label="Prediction Result")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
               view, condition, grade, sqft_above, sqft_basement, yr_built, 
               yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15, 
               sale_year, sale_month],
        outputs=output
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            [3, 2.5, 2000, 5000, 1, 0, 0, 3, 7, 1800, 200, 1990, 0, 98103, 47.65, -122.35, 2200, 4800, 2023, 10],
            [4, 3.0, 3000, 7000, 2, 1, 3, 4, 9, 2500, 500, 2000, 0, 98112, 47.63, -122.30, 3100, 7200, 2023, 6],
        ],
        inputs=[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
               view, condition, grade, sqft_above, sqft_basement, yr_built, 
               yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15, 
               sale_year, sale_month],
        label="Example Properties"
    )

if __name__ == "__main__":
    demo.launch(share=True)


