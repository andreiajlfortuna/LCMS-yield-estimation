import os
import pandas as pd
import glob
import csv
from typing import Optional, Tuple, List
from enum import Enum
from scipy.signal import find_peaks
from multiprocessing import Pool

class OutputFile(Enum):
    STD = "peak_detection_results.csv"      # Output for standard peak detection results
    RXN = "peak_detection_reaction.csv"     # Output for reaction peak detection results
    YIELD = "reaction_yield.csv"             # Output for calculated reaction yields

# Main class to handle peak detection and yield calculation
class PeakDetector:
    def __init__(
        self,
        data_folder: str = "data",            # Folder where input CSV files are stored
        target_wavelength: float = 254.0,    # Target wavelength for peak detection
        retention_window: float = 0.02,      # Time window around expected peak time to search for peaks
        min_peaks_required: int = 2,         # Minimum number of peaks required in standard file
        height_threshold: float = 1.0,       # Minimum peak height threshold for detection
        peak_distance: int = 5                # Minimum distance between peaks to be considered separate
    ):
        # Store parameters as instance variables
        self.data_folder = data_folder
        self.target_wavelength = target_wavelength
        self.retention_window = retention_window
        self.min_peaks_required = min_peaks_required
        self.height_threshold = height_threshold
        self.peak_distance = peak_distance

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove any leading/trailing whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def find_closest_wavelength(self, wavelengths: List[float]) -> float:
        # Find the wavelength column closest to the target wavelength
        return min(wavelengths, key=lambda x: abs(x - self.target_wavelength))

    def extract_peaks(self, series: pd.Series) -> List[int]:
        # Use scipy's find_peaks to locate peaks in the signal series based on height and distance
        return find_peaks(series, height=self.height_threshold, distance=self.peak_distance)[0].tolist()

    def get_peak_near(
        self, df: pd.DataFrame, wavelength_col: str, time_col: str, target_time: float
    ) -> Tuple[Optional[float], Optional[float]]:
        # Find peak near a target retention time within the specified retention window
        mask = df[time_col].between(target_time - self.retention_window, target_time + self.retention_window)
        local = df[mask]
        if local.empty:
            # Return None if no data points in window
            return None, None
        spectrum = local[wavelength_col].values
        times = local[time_col].values
        peaks = self.extract_peaks(pd.Series(spectrum))
        if not peaks:
            # Return None if no peaks are found in the window
            return None, None
        # Select the peak with the highest intensity within the window
        peak_idx = peaks[pd.Series(spectrum)[peaks].argmax()]
        # Return the retention time and intensity of the peak, rounded for neatness
        return round(times[peak_idx], 4), round(spectrum[peak_idx], 2)

    def detect_std_peaks(self, file_path: str) -> Optional[dict]:
        # Detect standard peaks from the standard PDA CSV file
        df = pd.read_csv(file_path)
        df = self.clean_column_names(df)

        # Assume the first column is time, the rest are wavelengths
        time_col = df.columns[0]
        wavelengths = [float(c) for c in df.columns[1:] if c.replace('.', '', 1).isdigit()]
        closest_wl = self.find_closest_wavelength(wavelengths)
        closest_wl_col = str(closest_wl)

        # Extract signal for the closest wavelength and find peaks
        spectrum = df[closest_wl_col].values
        peaks = self.extract_peaks(pd.Series(spectrum))
        if len(peaks) < self.min_peaks_required:
            # Not enough peaks to continue
            return None

        # Find the two highest peaks (intensity-wise)
        intensities = spectrum[peaks]
        top2_idx = [x for _, x in sorted(zip(intensities, peaks), reverse=True)[:2]]
        times = df[time_col].values
        # Return a dictionary with peak times and intensities for the compound and caffeine
        return {
            "compound_time": round(times[top2_idx[0]], 4),
            "compound_intensity": round(spectrum[top2_idx[0]], 2),
            "caffeine_time": round(times[top2_idx[1]], 4),
            "caffeine_intensity": round(spectrum[top2_idx[1]], 2),
            "closest_wl": closest_wl_col,
            "time_col": time_col
        }

    def process_file(self, file_path: str) -> Tuple[Optional[List], Optional[List]]:
        # Process a single standard file and its corresponding reaction file
        reaction_id = os.path.basename(file_path).split("_")[0]
        std_peaks = self.detect_std_peaks(file_path)
        if not std_peaks:
            print(f"{reaction_id}: Not enough peaks in STD file.")
            return None, None

        # Prepare result row for standard peaks
        std_result = [
            reaction_id,
            std_peaks["compound_time"],
            std_peaks["compound_intensity"],
            std_peaks["caffeine_time"],
            std_peaks["caffeine_intensity"]
        ]

        # Construct corresponding reaction file path
        rxn_file = os.path.join(self.data_folder, f"{reaction_id}_PDA.csv")
        if not os.path.exists(rxn_file):
            print(f"{reaction_id}: Reaction file missing.")
            return std_result, None

        try:
            # Load reaction file and clean columns
            df_rxn = pd.read_csv(rxn_file)
            df_rxn = self.clean_column_names(df_rxn)
            wl_col = std_peaks["closest_wl"]
            time_col = std_peaks["time_col"]

            if wl_col not in df_rxn.columns:
                print(f"{reaction_id}: Wavelength {wl_col} not found in reaction file.")
                return std_result, None

            # Find peaks near the standard compound and caffeine retention times in reaction file
            comp_time, comp_int = self.get_peak_near(df_rxn, wl_col, time_col, std_peaks["compound_time"])
            caf_time, caf_int = self.get_peak_near(df_rxn, wl_col, time_col, std_peaks["caffeine_time"])

            # Prepare reaction result row
            rxn_result = [reaction_id, comp_time, comp_int, caf_time, caf_int]
            return std_result, rxn_result

        except Exception as e:
            # Catch any exceptions during reaction file processing and report
            print(f"{reaction_id}: Error in reaction file → {e}")
            return std_result, None

    def calculate_yields(
        self,
        results_std: List[List],
        results_rxn: List[List],
        output_file: str = OutputFile.YIELD.value
    ) -> List[List]:
        # Calculate yields based on standard and reaction peak intensities
        try:
            # Convert results lists to DataFrames for easier manipulation
            df_std = pd.DataFrame(results_std, columns=[
                "reaction_id", "compound_retention_time", "compound_max_intensity",
                "caffeine_retention_time", "caffeine_max_intensity"])
            df_rxn = pd.DataFrame(results_rxn, columns=[
                "reaction_id", "compound_retention_time", "compound_max_intensity",
                "caffeine_retention_time", "caffeine_max_intensity"])

            # Merge standard and reaction data on reaction_id
            df_merged = pd.merge(df_std, df_rxn, on="reaction_id", suffixes=('_std', '_rxn'))
            yield_results = []

            for _, row in df_merged.iterrows():
                try:
                    # Extract peak intensities from the standard and reaction
                    Ipmax, Icaffeinemax = row["compound_max_intensity_std"], row["caffeine_max_intensity_std"]
                    Ip, Icaffeine = row["compound_max_intensity_rxn"], row["caffeine_max_intensity_rxn"]
                    # Check for invalid or zero values to avoid division errors
                    if any(pd.isna([Ipmax, Icaffeinemax, Ip, Icaffeine])) or Icaffeine == 0 or Icaffeinemax == 0:
                        raise ValueError("Invalid intensity data")
                    # Calculate yield as normalized ratio of compound intensity to caffeine intensity
                    y = (Ip / Icaffeine) / (Ipmax / Icaffeinemax)
                    yield_results.append([row["reaction_id"], f"{round(y * 100, 2):.2f}"])
                except Exception:
                    # If an error occurs, append zero yield
                    yield_results.append([row["reaction_id"], "0.00"])

            # Save yield results to CSV
            pd.DataFrame(yield_results, columns=["reaction_id", "yield (%)"]).to_csv(output_file, index=False)
            print(f"Yield results saved to '{output_file}'")
            return yield_results
        except Exception as e:
            print(f"Error calculating yields: {e}")
            return []

    def process_all(self):
        # Main driver method to process all standard files in data folder
        file_paths = glob.glob(os.path.join(self.data_folder, "*_std_PDA.csv"))
        # Use multiprocessing Pool to parallelize processing files
        with Pool() as pool:
            results = pool.map(self.process_file, file_paths)

        # Separate results into standard and reaction results, filtering out None entries
        results_std = [r[0] for r in results if r[0] is not None]
        results_rxn = [r[1] for r in results if r[1] is not None]

        # Define output column names
        std_columns = [
            "reaction_id", "compound_retention_time", "compound_max_intensity",
            "caffeine_retention_time", "caffeine_max_intensity"
        ]
        rxn_columns = std_columns  # Same columns for reaction data

        # Save standard peak detection results
        pd.DataFrame(results_std, columns=std_columns).to_csv(OutputFile.STD.value, index=False)
        print(f"Standard peak detection results saved to '{OutputFile.STD.value}'")

        # Save reaction peak detection results
        pd.DataFrame(results_rxn, columns=rxn_columns).to_csv(OutputFile.RXN.value, index=False)
        print(f"Reaction peak detection results saved to '{OutputFile.RXN.value}'")

        # Calculate and save yields based on peak data
        self.calculate_yields(results_std, results_rxn)


if __name__ == "__main__": 
    detector = PeakDetector()
    detector.process_all()
