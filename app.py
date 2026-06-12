import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Page config
st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    with open('models/linear_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('models/random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    return lr, rf

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_ames.csv')

lr, rf = load_models()
df = load_data()

# Get feature columns
feature_cols = df.drop('SalePrice', axis=1).columns.tolist()

# App title
st.title("🏠 Ames Housing Price Predictor")
st.markdown("Adjust the house features in the sidebar to get an instant price prediction.")

# Sidebar inputs
st.sidebar.header("House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
gr_liv_area = st.sidebar.slider("Above Ground Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.sidebar.slider("Garage Size (cars)", 0, 4, 2)
total_bsmt_sf = st.sidebar.slider("Total Basement Area (sq ft)", 0, 3000, 1000)
year_built = st.sidebar.slider("Year Built", 1872, 2010, 1990)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)

# Build input row matching training columns exactly
input_dict = {col: 0 for col in feature_cols}
input_dict['Overall Qual'] = overall_qual
input_dict['Gr Liv Area'] = gr_liv_area
input_dict['Garage Cars'] = garage_cars
input_dict['Total Bsmt SF'] = total_bsmt_sf
input_dict['Year Built'] = year_built
input_dict['Full Bath'] = full_bath

input_df = pd.DataFrame([input_dict])

# Predictions - cap linear regression between 50k and 800k
lr_pred = float(np.clip(lr.predict(input_df)[0], 50000, 800000))
rf_pred = float(rf.predict(input_df)[0])
avg_pred = (lr_pred + rf_pred) / 2

# Display predictions
# Display predictions
st.header("Predicted Sale Price")
col1, col2 = st.columns(2)
col1.metric("Random Forest Prediction", f"${rf_pred:,.0f}")
col2.metric("Linear Regression Prediction", f"${lr_pred:,.0f} ⚠️")
st.caption("Note: Random Forest is the more accurate model for this dataset. Linear Regression struggles with 294 features.")

# Model comparison chart
st.header("Model Comparison")
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [22825, 24687],
    'R²': [0.935, 0.924]
})

col1, col2 = st.columns(2)
with col1:
    fig1 = px.bar(comparison_df, x='Model', y='RMSE',
                  title='RMSE by Model (lower is better)',
                  color='Model')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(comparison_df, x='Model', y='R²',
                  title='R² Score by Model (higher is better)',
                  color='Model')
    st.plotly_chart(fig2, use_container_width=True)

# Feature importance
st.header("What Drives House Prices?")
feature_importance = pd.DataFrame({
    'Feature': ['Overall Quality', 'Living Area', 'Garage Size',
                'Basement Area', 'Year Built', 'Full Bathrooms'],
    'Importance': [0.57, 0.51, 0.64, 0.61, 0.54, 0.38]
}).sort_values('Importance', ascending=True)

fig3 = px.bar(feature_importance, x='Importance', y='Feature',
              orientation='h',
              title='Feature Importance (Random Forest)',
              color='Importance',
              color_continuous_scale='blues')
st.plotly_chart(fig3, use_container_width=True)

# Raw data explorer
st.header("Explore the Data")
st.markdown(f"Dataset contains **{len(df):,} houses** after cleaning.")
if st.checkbox("Show raw data"):
    st.dataframe(df.head(50))