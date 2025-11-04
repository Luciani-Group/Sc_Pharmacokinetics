import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_concentration(F, D, V, Cl, ka, time):
    """
    Calculate plasma concentration C(t) using the one-compartment PK model with first-order absorption.

    Parameters:
    - F: Bioavailability (fraction)
    - D: Dose administered (mg)
    - V: Volume of distribution (L)
    - Cl: Clearance (L/h)
    - ka: Absorption rate constant (1/h)
    - time: Array of time points (h)

    Returns:
    - C_t: Concentration at each time point (mg/L)
    """
    ke = Cl / V  # Elimination rate constant (1/h)

    # Handle the case where ka == ke to avoid division by zero
    if ka == ke:
        raise ValueError("ka and ke are equal, leading to division by zero in the concentration equation.")

    C_t = (F * D * ka) / (V * (ka - ke)) * (np.exp(-ke * time) - np.exp(-ka * time))
    return C_t

def plot_concentration_time(df, output_path='concentration_time_plots.png'):
    """
    Plot the concentration-time profiles for each drug in the dataframe.

    Parameters:
    - df: pandas DataFrame containing Dose, V, Cl, ka, Drug columns
    - output_path: File path to save the plot
    """
    F = 1  # 100% bioavailability
    time = np.linspace(0, 24, 100)  # Time from 0 to 24 hours

    plt.figure(figsize=(12, 8))

    for index, row in df.iterrows():
        D = row['Dose']
        V = row['V']
        Cl = row['Cl']
        ka = row['ka']
        drug_name = row['Drug'] if 'Drug' in row else f"Drug {index + 1}"

        try:
            C_t = calculate_concentration(F, D, V, Cl, ka, time)
        except ValueError as e:
            print(f"Error for {drug_name} at index {index}: {e}")
            continue

        plt.plot(time, C_t, label=drug_name)

    plt.title('One-Compartment PK Model Concentration-Time Profiles')
    plt.xlabel('Time (hours)')
    plt.ylabel('Plasma Concentration (mg/L)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    print(f"Concentration-time plots saved to '{output_path}'.")

    # Show the plot
    plt.show()

def main():
    # File paths
    predictions_input_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions_with_PK.xlsx'
    predictions_output_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\Predictions_with_PK_with_Ct.xlsx'
    concentration_plot_path = r'C:\Users\remoe\OneDrive - Universitaet Bern\PhD - Remo\AI_Sc_Model\Model\concentration_time_plots.png'

    # Load the predicted data
    try:
        df = pd.read_excel(predictions_input_path)
        print("Predicted data loaded successfully.")
    except FileNotFoundError:
        print(f"Predictions file not found at path: {predictions_input_path}")
        return

    # Check for required columns
    required_columns = ['Dose', 'V', 'Cl', 'ka', 'Drug']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns in the predictions file: {missing_columns}")
        return

    # Handle missing values
    initial_shape = df.shape
    df_clean = df.dropna(subset=required_columns)
    final_shape = df_clean.shape
    dropped_rows = initial_shape[0] - final_shape[0]
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing values in required columns.")

    if df_clean.empty:
        print("No data available for plotting after handling missing values.")
        return

    # Plot concentration-time profiles
    plot_concentration_time(df_clean, output_path=concentration_plot_path)

    # Optionally, save the dataframe with additional C(t) data
    # For example, saving concentrations at specific time points
    # Here, we skip this step as it's not requested

if __name__ == "__main__":
    main()
