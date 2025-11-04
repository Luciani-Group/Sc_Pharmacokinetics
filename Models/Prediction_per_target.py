import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# File paths
predictions_input_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions.xlsx'
scaler_load_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\models_per_target\scaler.pkl'
model_dir = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\models_per_target'
predictions_output_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions_with_PK.xlsx'

# Target list
target_columns = ['V', 'Cl', 'ka']
log_transformed = ['Cl', 'ka']  # These were log-transformed during training

# Feature columns (must match training)
feature_columns = [
    'Dose',            # still include dose
    'MolWt',
    'LogP',
    'TPSA',
    'NumHDonors',
    'NumHAcceptors',
    'RingCount',
]



# Load scaler
try:
    scaler = joblib.load(scaler_load_path)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Scaler file not found at path: {scaler_load_path}")
    exit()

# Load input data
try:
    new_data = pd.read_excel(predictions_input_path)
    print("New data loaded successfully.")
except FileNotFoundError:
    print(f"Predictions file not found at path: {predictions_input_path}")
    exit()

# Drop rows with missing features
X_new = new_data[feature_columns].copy()
X_new = X_new.dropna()
new_data = new_data.loc[X_new.index]  # Keep only valid rows

# Scale features
X_new_scaled = scaler.transform(X_new)

# Predict for each target separately
for target in target_columns:
    model_path = f"{model_dir}/model_{target}.pkl"
    try:
        model = joblib.load(model_path)
        print(f"Model for {target} loaded successfully.")
    except FileNotFoundError:
        print(f"Model for {target} not found at: {model_path}")
        exit()

    pred = model.predict(X_new_scaled)
    
    # Reverse log if needed
    if target in log_transformed:
        pred = np.expm1(pred)

    new_data[target] = pred  # Add to DataFrame

# Save predictions
new_data.to_excel(predictions_output_path, index=False)
print(f"Predictions saved to: {predictions_output_path}")
