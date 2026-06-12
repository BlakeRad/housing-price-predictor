import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load cleaned data
df = pd.read_csv('../data/cleaned_ames.csv')

# Separate features from target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate both models
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name}:")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R²:   {r2:.3f}")
    print()

evaluate(lr, X_test, y_test, "Linear Regression")
evaluate(rf, X_test, y_test, "Random Forest")

# Save models
os.makedirs('../models', exist_ok=True)
with open('../models/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('../models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Models saved!")
