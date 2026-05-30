import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# === CONFIG ===
DATASET_PATH = r"C:\Users\AMMAR HAZIQ\Desktop\DigitalTwin_Project\final_integrated_data.csv"
MODEL_OUTPUT_PATH = r"C:\Users\AMMAR HAZIQ\Desktop\DigitalTwin_Project\ev_range_model.pkl"

print("[INFO] Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("[INFO] Dataset shape:", df.shape)
print("[INFO] Columns:", df.columns.tolist())

# === Drop unnecessary columns ===
df = df.drop(columns=["source_url"], errors="ignore")

# Drop missing target values
df = df.dropna(subset=["range_km"])

# === Identify numeric & categorical columns ===
# === Identify numeric & categorical columns ===
# MOVED: "cargo_volume_l" is now a numeric feature
numeric_cols = [
    "top_speed_kmh", "battery_capacity_kWh", "number_of_cells",
    "torque_nm", "efficiency_wh_per_km", "acceleration_0_100_s",
    "fast_charging_power_kw_dc", "towing_capacity_kg", "seats",
    "length_mm", "width_mm", "height_mm", "cargo_volume_l", "traffic_density", "inclination" 
]

categorical_cols = [
    "brand", "model", "battery_type", "fast_charge_port",
    "drivetrain", "segment", "car_body_type"
]

# === Robust Type Casting & Missing Value Handling ===
# 1. Handle numeric columns: force to numeric, then fill NaNs with 0
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 2. Handle categorical columns: fill NaNs with "Unknown", then force to string
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

# The two loops above have safely handled all missing values.
# === Features & target ===
X = df[numeric_cols + categorical_cols]
y = df["range_km"]

# === Preprocessor ===
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# === Model Pipeline ===
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    ))
])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[INFO] Training model...")
model.fit(X_train, y_train)
print("[INFO] Model training complete!")

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n[RESULTS]")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# === Save model ===
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"\n✅ Model saved successfully at: {MODEL_OUTPUT_PATH}")
