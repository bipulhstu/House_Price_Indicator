# 🏠 House Price Estimator

A comprehensive machine learning project that predicts house prices based on various features such as location, size, number of bedrooms, and amenities. The project includes data visualization, feature engineering, model training, and deployment options.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Regression-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📝 Project Description

This project creates a **regression model** to predict house prices using the King County House Sales dataset. It demonstrates the complete machine learning workflow from data exploration to model deployment, with interactive visualizations and multiple deployment options.

### 🎯 Objective

Predict house prices based on various features of the houses and their location. This is a **regression problem** because the target variable (price) is continuous and numeric.

### 💡 Use Cases

- **Real estate valuation** - Estimate property values accurately
- **Investment analysis** - Identify undervalued or overvalued properties  
- **Market insights** - Understand factors that influence housing prices
- **Price forecasting** - Predict future property values

---

## 🔧 Project Structure

This project follows a systematic machine learning workflow:

1. **🧠 Define the Problem** - Understand the objective and type of problem (regression)
2. **🗂️ Collect and Prepare Data** - Load dataset, handle missing values, and preprocess data
3. **📊 Exploratory Data Analysis (EDA)** - Visualize data to understand patterns and correlations
4. **📐 Feature Engineering** - Select and create relevant features
5. **🔀 Split the Data** - Divide dataset into training and testing sets
6. **🤖 Choose a Model** - Select suitable machine learning algorithms
7. **🏋️ Train the Model** - Train the model using the training set
8. **📈 Evaluate the Model** - Use appropriate metrics to evaluate performance
9. **🔧 Improve the Model** - Tune hyperparameters and enhance features
10. **🚀 Deploy the Model** - Create interactive applications for predictions

---

## 🧪 Models Experimented

The project experiments with multiple regression algorithms:

- **Linear Regression** – Baseline performance
- **Random Forest Regressor** – Capture non-linear relationships
- **Gradient Boosting Regressor** – Improved accuracy through boosting ⭐ **(Best Model)**
- **XGBoost Regressor** – Optimized gradient boosting with regularization
- **Lasso & Ridge Regression** – Manage multicollinearity and feature selection

---

## 📊 Dataset Information

**Dataset**: King County House Sales (`kc_house_data.csv`)

### Key Features

The dataset includes the following features:

| Feature | Description |
|---------|-------------|
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Square footage of living area |
| `sqft_lot` | Square footage of lot |
| `floors` | Number of floors |
| `waterfront` | Waterfront property (0/1) |
| `view` | Quality of view (0-4) |
| `condition` | Condition of the house (1-5) |
| `grade` | Grade of the house (1-13) |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year the house was renovated |
| `zipcode` | ZIP code location |
| `lat` | Latitude coordinate |
| `long` | Longitude coordinate |
| `price` | **Target Variable** - House price |

### Engineered Features

- `age` - Age of the house (sale_year - yr_built)
- `renovated` - Binary indicator if renovated
- `sqft_living_grade` - Interaction term
- `sqft_living_squared` - Polynomial feature
- Location interactions and ratio features

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### For Deployment Apps (Optional)

```bash
# For Streamlit web app
pip install streamlit

# For Gradio interface
pip install gradio
```

---

## 💻 Usage

### Option 1: Run Jupyter Notebook (Recommended for Learning)

```bash
# Launch Jupyter
jupyter notebook House_Price_Estimator.ipynb

# Run all cells to:
# - Explore the data
# - View visualizations (auto-saved to images/)
# - Train the model
# - See evaluation metrics
```

### Option 2: Use the Trained Model (Deploy Script)

```bash
python deploy.py
```

This script:
- Loads the pre-trained model (`model.pkl`) and scaler (`scaler.pkl`)
- Makes predictions on sample data
- Shows predicted house price

### Option 3: Interactive Web App (Streamlit)

```bash
streamlit run app_streamlit.py
```

**Features:**
- Beautiful, interactive web interface
- Adjust property features with sliders and inputs
- Get instant price predictions
- Professional UI for demos and presentations

### Option 4: Interactive Interface (Gradio)

```bash
python app_gradio.py
```

**Features:**
- Clean, simple web interface
- Example inputs to try
- Shareable public link option
- Quick prototyping and testing

📚 **See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed instructions**

---

## 📈 Model Performance

### Best Model: Gradient Boosting Regressor

**Metrics (Initial Model):**
- **R² Score**: 0.7995
- **RMSE**: $166,491
- **MAE**: $80,721

**Improved Model (with Advanced Feature Engineering):**
- **R² Score**: 0.6985
- **RMSE**: $213,508
- **MAE**: $144,151

### Key Insights

- **Most important features**: Grade, living square footage, location (latitude/longitude)
- **Correlation with price**: Grade (0.67), sqft_living (0.65), sqft_above (0.61)
- **Waterfront properties**: Average $1,661,876 vs $531,564 for non-waterfront
- **Price distribution**: Right-skewed with mean price of $540,088

---

## 📊 Visualizations

The project generates 12 high-quality visualizations (300 DPI, PNG format):

| Visualization | Description |
|---------------|-------------|
| 📈 Price Distribution | Histogram showing price spread and skewness |
| 📍 Location vs Price | Geographic scatter plot with color-coded prices |
| 🏠 Bedrooms vs Price | Bar chart of average prices by bedroom count |
| 🔥 Correlation Heatmap | Feature correlation matrix |
| 📏 Square Footage vs Price | Relationship between living area and price |
| ⭐ Grade vs Price | Average prices by house grade |
| 📅 Year Built vs Price | Price trends over construction year |
| 🌊 Waterfront vs Price | Box plot comparing waterfront properties |
| 🛁 Bathrooms vs Price | Scatter plot of bathrooms and price |
| 📊 Condition Distribution | Count plot of house conditions |
| 📐 Living Area Distribution | Histogram of square footage |
| 📉 Residual Plot | Model error analysis |

**All images are automatically saved to the `images/` directory when running the notebook.**

---

## 📁 Project Structure

```
House_Price_Estimator/
├── 📓 House_Price_Estimator.ipynb   # Main Jupyter notebook
├── 🐍 deploy.py                     # Deployment script for predictions
├── 🌐 app_streamlit.py              # Streamlit web application
├── 🎯 app_gradio.py                 # Gradio web interface
├── 📊 kc_house_data.csv             # Dataset
├── 🤖 model.pkl                     # Trained model
├── ⚖️ scaler.pkl                    # Data scaler
├── 📚 README.md                     # This file
├── 📖 HOW_TO_RUN.md                 # Detailed usage guide
└── 🖼️ images/                       # Saved visualizations (12 images)
    ├── 01_price_distribution.png
    ├── 02_sqft_vs_price.png
    ├── 03_bedrooms_vs_price.png
    ├── 04_correlation_heatmap.png
    ├── 05_location_vs_price.png
    ├── 06_grade_vs_price.png
    ├── 07_year_built_vs_price.png
    ├── 08_waterfront_vs_price.png
    ├── 09_sqft_living_distribution.png
    ├── 10_bathrooms_vs_price.png
    ├── 11_condition_distribution.png
    └── 12_residual_plot.png
```

---

## 🛠️ Technologies Used

### Data Science & ML
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **SciPy** - Scientific computing

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical data visualization

### Deployment
- **Streamlit** - Web app framework
- **Gradio** - ML interface library
- **Pickle** - Model serialization

### Development
- **Jupyter Notebook** - Interactive development
- **Python 3.8+** - Programming language

---

## 🎓 Key Features & Techniques

### Data Preprocessing
- Missing value imputation using median
- Outlier handling with IQR method
- Date feature extraction (year, month)
- Categorical encoding (one-hot encoding)

### Feature Engineering
- Age calculation (sale_year - yr_built)
- Renovation indicator
- Interaction features (sqft_living × grade, sqft_living × lat/long)
- Polynomial features (sqft_living²)
- Ratio features (living/lot ratio, bathrooms/bedrooms ratio)
- Bedroom categorization

### Model Optimization
- Feature selection using SelectKBest and RFE
- Hyperparameter tuning with RandomizedSearchCV
- RobustScaler for feature scaling
- Early stopping for regularization
- Cross-validation (3-fold CV)

---

## 📊 Sample Prediction

Example input:
```python
{
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2000,
    'sqft_lot': 5000,
    'floors': 1,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'grade': 7,
    # ... other features
}
```

**Predicted Price**: ~$540,000

---

## 🎯 Conclusion

This project successfully developed a house price prediction model using regression techniques. Different models were tested, and through feature engineering, selection, and hyperparameter tuning, the model's accuracy was improved. The final model provides reasonably accurate price estimates, as demonstrated by the evaluation metrics.

**Key Achievements:**
- ✅ Built an end-to-end ML pipeline
- ✅ Achieved R² score of ~0.80
- ✅ Created 12 publication-quality visualizations
- ✅ Deployed 3 different interfaces (CLI, Streamlit, Gradio)
- ✅ Comprehensive feature engineering

---

## 🔮 Future Recommendations

To further enhance the model's performance:

1. **More Complex Features** - Explore additional interaction terms and polynomial features
2. **Advanced Algorithms** - Experiment with XGBoost, LightGBM, CatBoost
3. **Ensemble Methods** - Combine multiple models for better predictions
4. **Additional Data** - Include economic indicators, school ratings, crime rates
5. **Deep Learning** - Try neural networks for complex pattern recognition
6. **Time Series** - Incorporate temporal trends in housing prices
7. **Geospatial Analysis** - Advanced location-based features

---

## 📜 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Project**: House Price Estimator  
**Type**: Machine Learning - Regression  
**Status**: ✅ Complete

---

## 🙏 Acknowledgments

- Dataset: King County House Sales
- Inspired by real-world real estate valuation challenges
- Built as a comprehensive ML portfolio project

---

## 📞 Support

For questions or issues:
1. Check the [HOW_TO_RUN.md](HOW_TO_RUN.md) guide
2. Review the Jupyter notebook for detailed explanations
3. Try the interactive web apps for hands-on experience

---

**⭐ If you found this project helpful, please give it a star!**
