# 🏠 Ames Housing Price Predictor

A machine learning project that predicts home sale prices in Ames, Iowa. 
Built this to practice data cleaning, feature engineering, and ML modeling 
with a real-world dataset.

## What it does
- Takes house features like square footage, quality rating, and year built
- Predicts the sale price using a Random Forest model
- Interactive Streamlit app with sliders to adjust features and see predictions update live

## Results
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $22,825 | 0.935 |
| Random Forest | $24,687 | 0.924 |

Random Forest ended up being the more reliable model — Linear Regression 
struggled with the 294 features after one-hot encoding.

## Tech used
- Python, Pandas, NumPy
- scikit-learn (Linear Regression, Random Forest)
- Plotly for charts
- Streamlit for the web app

## How to run it yourself
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Download the [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) and put it in `data/`
4. Run `python src/model_train.py` to train the models
5. Run `streamlit run app.py` to launch the app

## Project structure
```
housing-price-predictor/
├── data/               # Dataset (not included, download from Kaggle)
├── models/             # Saved model files (generated when you run model_train.py)
├── src/
│   └── model_train.py  # Training script
├── app.py              # Streamlit app
└── requirements.txt
```
