import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# File paths
predictions_input_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions.xlsx'
model_load_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\multi_target_xgboost_model.pkl'
scaler_load_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\scaler.pkl'
predictions_output_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions_with_PK.xlsx'

# Load the trained model and scaler
try:
    multi_target_model = joblib.load(model_load_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at path: {model_load_path}")
    exit()

try:
    scaler = joblib.load(scaler_load_path)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Scaler file not found at path: {scaler_load_path}")
    exit()

# Load the new data from the Predictions.xlsx file
try:
    new_data = pd.read_excel(predictions_input_path)
    print("New data loaded successfully.")
except FileNotFoundError:
    print(f"Predictions file not found at path: {predictions_input_path}")
    exit()

# Specify the feature columns (must match those used in training)
feature_columns = ['Dose', 'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']

# Check if all required feature columns exist
missing_features = [col for col in feature_columns if col not in new_data.columns]
if missing_features:
    print(f"Missing feature columns in the new data: {missing_features}")
    exit()

# Extract features
X_new = new_data[feature_columns]

# Handle any missing values (depending on your preference)
# Here, we'll drop rows with missing values. Alternatively, you can impute them.
initial_shape = X_new.shape
X_new = X_new.dropna()
final_shape = X_new.shape
dropped_rows = initial_shape[0] - final_shape[0]
if dropped_rows > 0:
    print(f"Dropped {dropped_rows} rows due to missing feature values.")

# If rows were dropped, update the original dataframe accordingly
new_data = new_data.loc[X_new.index]

# Feature Scaling using the loaded scaler
X_new_scaled = scaler.transform(X_new)

# Make predictions using the loaded model
y_pred = multi_target_model.predict(X_new_scaled)

# Assign predictions to the DataFrame
# Ensure that the order of targets matches ['V', 'Cl', 'ka']
new_data['V'] = y_pred[:, 0]
new_data['Cl'] = y_pred[:, 1]
new_data['ka'] = y_pred[:, 2]

# Save the predictions to a new Excel file
new_data.to_excel(predictions_output_path, index=False)
print(f"Predictions have been made and saved to '{predictions_output_path}'.")
