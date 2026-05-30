import pandas as pd
import joblib
import time
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import firebase_admin
from firebase_admin import credentials, firestore

# =============================
# CONFIG
# =============================
MODEL_PATH = r"C:\Users\AMMAR HAZIQ\Desktop\DigitalTwin_Project\ev_range_model.pkl"
DATA_PATH = r"C:\Users\AMMAR HAZIQ\Desktop\DigitalTwin_Project\final_integrated_data.csv"

OUTPUT_DIR = r"C:\evdata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PRED_PATH = os.path.join(OUTPUT_DIR, "predicted_range_live.txt")
OUTPUT_ACTUAL_PATH = os.path.join(OUTPUT_DIR, "actual_range_live.txt")
OUTPUT_BATTERY = os.path.join(OUTPUT_DIR, "battery_live.txt")
OUTPUT_SOH = os.path.join(OUTPUT_DIR, "soh_live.txt")
OUTPUT_TEMP = os.path.join(OUTPUT_DIR, "temp_live.txt")

SERVICE_ACCOUNT_PATH =r"C:\Users\AMMAR HAZIQ\Desktop\DigitalTwin_Project\digital-twin-g7-project-firebase-adminsdk-fbsvc-d81d9f71eb.json"

# =============================
# Firebase
# =============================
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# =============================
# Load model + dataset
# =============================
print("[INFO] Loading trained model...")
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {len(df)} rows")

# Save original values before cleaning
actual_range_list = df["range_km"].copy()
battery_percent_list = df["battery_percent"].copy()
battery_temperature_list = df["battery_temperature"].copy()   # <-- ADDED

# Remove columns not used by model (BUT KEEP temperature)
df = df.drop(columns=["timestamp"], errors="ignore")
df = df.drop(columns=["range_km"], errors="ignore")
# df = df.drop(columns=["battery_temperature"], errors="ignore")   # <-- REMOVED

numeric_cols = [
    "top_speed_kmh", "battery_capacity_kWh", "number_of_cells",
    "torque_nm", "efficiency_wh_per_km", "acceleration_0_100_s",
    "fast_charging_power_kw_dc", "towing_capacity_kg", "seats",
    "length_mm", "width_mm", "height_mm", "battery_percent"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = [
    "brand", "model", "battery_type", "fast_charge_port",
    "cargo_volume_l", "drivetrain", "segment", "car_body_type"
]
for col in categorical_cols:
    df[col] = df[col].astype(str)

# =============================
# Live Simulation Loop
# =============================
print("[INFO] Starting live prediction simulation...")

pred_list = []
actual_list = []

for idx, row in df.iterrows():
    sample = row.to_frame().T

    # Predict
    try:
        predicted_range = model.predict(sample)[0]
    except Exception as e:
        print(f"[WARN] Skipped row due to error: {e}")
        continue

    actual_range = actual_range_list.iloc[idx]
    battery_percent = battery_percent_list.iloc[idx]
    battery_temperature = battery_temperature_list.iloc[idx]     # <-- NEW

    pred_list.append(predicted_range)
    actual_list.append(actual_range)

    # =============================
    # SOH ESTIMATION
    # =============================
    error = abs(predicted_range - actual_range)
    soh_drop = error * 0.02
    current_soh = max(0, 100 - soh_drop)

    # =============================
    # Write TXT files
    # =============================
    with open(OUTPUT_PRED_PATH, "w") as f:
        f.write(f"Predicted Range = {predicted_range:.2f} km\n")

    with open(OUTPUT_ACTUAL_PATH, "w") as f:
        f.write(f"Actual Range = {actual_range:.2f} km\n")

    with open(OUTPUT_BATTERY, "w") as f:
        f.write(f"Battery = {battery_percent:.2f} %\n")

    with open(OUTPUT_SOH, "w") as f:
        f.write(f"{current_soh:.2f}")

    with open(OUTPUT_TEMP, "w") as f:                       # <-- NEW
        f.write(f"Battery Temperature = {battery_temperature:.2f} Celcius\n")

    # =============================
    # Upload to Firebase
    # =============================
    try:
        db.collection("ev_live").document("current").set({
            "predicted_range": float(predicted_range),
            "actual_range": float(actual_range),
            "battery_percent": float(battery_percent),
            "battery_temperature": float(battery_temperature),   # <-- NEW
            "soh": float(current_soh),
            "timestamp": firestore.SERVER_TIMESTAMP
        })

    except Exception as e:
        print(f"[ERROR] Firebase upload failed at idx {idx}: {e}")

    print(
        f"[UPDATE] Row {idx}: Batt={battery_percent:.1f}% | Temp={battery_temperature:.1f}°C | "
        f"Pred={predicted_range:.1f} km | Actual={actual_range:.1f} km | SOH={current_soh:.2f}%"
    )

    time.sleep(1)

# =============================
# RMSE
# =============================
rmse = np.sqrt(mean_squared_error(actual_list, pred_list))
print(f"\n[INFO] Simulation complete! RMSE = {rmse:.4f} km")
