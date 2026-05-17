import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
 
df = pd.read_csv("insurance.csv")
print(df.head())
print(df.info())       # column types
print(df.describe())   # basic stats
 
# Convert smoker (yes/no text) to 1 o 0
df["smoker"] = (df["smoker"] == "yes").astype(int)
 
# life expectancy related predictors
X = df[["age", "bmi", "smoker", "children"]]
y = df["charges"] #<-- variable we want to predict
 
# 80% for training, 20% for testing (random_state fixes the shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 
# Training the model
model = LinearRegression()
model.fit(X_train, y_train)   #<-- learn from training data
 
y_pred = model.predict(X_test)
 
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
 
print(f"MAE : ${mae:,.0f}")    # average dollar error
print(f"R²  : {r2:.3f}")      # 1.0 = perfect, 0 = not any better than the mean
 
# View coefficients (see how much each predictor shifts the predicted charge)
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
print(coef_df)
print(f"Intercept: {model.intercept_:.2f}")
 
# Plot actual vs predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # perfect-fit line
plt.xlabel("Actual charges ($USD)")
plt.ylabel("Predicted charges ($USD)")
plt.title("Actual vs Predicted insurance charges" "\n"
" from Life expectancy indicators: Age BMI Smoking Children")
plt.tight_layout()
plt.show()

#Data Source: https://www.kaggle.com/datasets/mirichoi0218/insurance