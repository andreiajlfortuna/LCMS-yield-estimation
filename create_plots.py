import os
import pandas as pd
import matplotlib.pyplot as plt
import glob


DATA_FOLDER = "data"
PLOT_FOLDER = "plots"
TARGET_WAVELENGTH = "253.77"
PEAK_FILE = "peak_detection_standard.csv"

os.makedirs(PLOT_FOLDER, exist_ok=True)

# Load caffeine retention times from peak_detection_standard.csv 
peak_df = pd.read_csv(PEAK_FILE)
# Create a dictionary {reaction_id: caffeine_retention_time}
caffeine_times = dict(zip(peak_df.iloc[:, 0], peak_df.iloc[:, 3]))

# Get reaction IDs based on standard PDA files
std_files = glob.glob(os.path.join(DATA_FOLDER, "*_std_PDA.csv"))
reaction_ids = [os.path.basename(f).split("_")[0] for f in std_files]

for reaction_id in reaction_ids:
    std_file = os.path.join(DATA_FOLDER, f"{reaction_id}_std_PDA.csv")
    rxn_file = os.path.join(DATA_FOLDER, f"{reaction_id}_PDA.csv")

    if not os.path.isfile(std_file) or not os.path.isfile(rxn_file):
        print(f"Missing PDA files for {reaction_id}, skipping...")
        continue

    std_df = pd.read_csv(std_file)
    rxn_df = pd.read_csv(rxn_file)

    if TARGET_WAVELENGTH not in std_df.columns or TARGET_WAVELENGTH not in rxn_df.columns:
        print(f"Wavelength {TARGET_WAVELENGTH} not found for {reaction_id}, skipping...")
        continue

    # Find time of max intensity in the standard PDA at target wavelength
    max_idx = std_df[TARGET_WAVELENGTH].idxmax()
    max_time_uv = std_df.iloc[max_idx, 0]

    # Get caffeine retention time if available
    caffeine_rt = caffeine_times.get(reaction_id, None)

    # === Plot PDA ===
    plt.figure(figsize=(10, 6))
    plt.plot(std_df.iloc[:, 0], std_df[TARGET_WAVELENGTH], label="STD", color="blue")
    plt.plot(rxn_df.iloc[:, 0], rxn_df[TARGET_WAVELENGTH], label="Reaction", color="green")
    plt.axvline(x=max_time_uv, color="red", linestyle="--", label=f"Max STD @ {round(max_time_uv, 2)} min")
    if caffeine_rt is not None and not pd.isna(caffeine_rt):
        plt.axvline(x=caffeine_rt, color="orange", linestyle="--", label=f"Caffeine @ {round(caffeine_rt, 2)} min")
    plt.title(f"{reaction_id} – Intensity at {TARGET_WAVELENGTH} nm")
    plt.xlabel("Retention Time (min)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(PLOT_FOLDER, f"{reaction_id}_PDA_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"PDA Plot saved for {reaction_id} → {output_path}")
