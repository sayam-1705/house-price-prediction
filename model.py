# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Step 2: Load dataset ---
data_path = Path(__file__).resolve().parent / "data" / "Housing.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

df = pd.read_csv(data_path)
print("Shape:", df.shape)
print(df.head())

# --- Step 3: Select features ---
X = df[["area"]]  # input feature (sq ft)
y = df["price"]  # target variable

# --- Step 4: Train/Test split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 5: Train the model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
print(f"\nSlope (m):     {model.coef_[0]:,.2f}")
print(f"Intercept (b): {model.intercept_:,.2f}")
print(f"R² Score:      {r2_score(y_test, y_pred):.4f}")
print(f"MSE:           {mean_squared_error(y_test, y_pred):,.2f}")

# --- Step 7: Predict for a new house ---
new_area = pd.DataFrame({"area": [5000]})  # change this value to test
predicted = model.predict(new_area)
print(f"\nPredicted price for 5000 sq ft: ₹{predicted[0]:,.0f}")

# --- Step 8: Plot ---
plot_df = X_test.copy()
plot_df["actual"] = y_test.values
plot_df["predicted"] = y_pred
plot_df = plot_df.sort_values("area")

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_df["area"], plot_df["actual"], color="steelblue", alpha=0.6, label="Actual"
)
plt.plot(
    plot_df["area"],
    plot_df["predicted"],
    color="tomato",
    linewidth=2,
    label="Regression line",
)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Linear Regression — House Price Prediction")
plt.legend()
plt.tight_layout()

if "agg" in plt.get_backend().lower():
    plt.savefig("regression_plot.png", dpi=150)
else:
    plt.show()
