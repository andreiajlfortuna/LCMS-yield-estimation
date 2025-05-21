import os
import pandas as pd
import glob
import csv
from typing import Optional, Tuple, List
from enum import Enum
from scipy.signal import find_peaks
from multiprocessing import Pool

class OutputFile(Enum):
    STD = "peak_detection_results.csv"         # Output file for standard peaks
    RXN = "peak_detection_reaction.csv"        # Output file for reaction peaks
    YIELD = "reaction_yield.csv"               # Output file for calculated yields

# Main class for peak detection and yield calculation
class PeakDetector:
    def __init__(
        self,
        data_folder: str = "data",              # Folder containing the input CSV files
        target_wavelength: float = 254.0,       # Desired wavelength to extract chromatograms
        retention_window: float = 0.02,         # Time window to search around a peak
        min_peaks_required: int = 2,            # Minimum number of peaks required to continue processing
        height_threshold: float = 1.0,          # Minimum peak height to consider
        peak_distance: int = 5                  # Minimum distance between detected peaks
    ):
        self.data_folder = data_folder
        self.target_wavelength = target_wavelength
        self.retention_window = retention_window
        self.min_peaks_required = min_peaks_required
        self.height_threshold = height_threshold
        self.peak_distance = peak_distance

    # Utility to clean whitespace from column names
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # Finds the closest wavelength to the target among those in the file
    def find_closest_wavelength(self, wavelengths: List[float]) -> float:
        return min(wavelengths, key=lambda x: abs(x - self.target_wavelength))

    # Detect peaks in the provided spectrum
    def extract_peaks(self, series: pd.Series) -> List[int]:
        return find_peaks(series, height=self.height_threshold, distance=self.peak_distance)[0].tolist()

    # Looks for a peak in a small window around a specific time
    def get_peak_near(
        self, df: pd.DataFrame, wavelength_col: str, time_col: str, target_time: float
    ) -> Tuple[Optional[float], Optional[float]]:
        mask = df[time_col].between(target_time - self.retention_window, target_time + self.retention_window)
        local = df[mask]
        if local.empty:
            return None, None
        spectrum = local[wavelength_col].values
        times = local[time_col].values
        peaks = self.extract_peaks(pd.Series(spectrum))
        if not peaks:
            return None, None
        # Find the peak with maximum intensity in the window
        peak_idx = peaks[pd.Series(spectrum)[peaks].argmax()]
        return round(times[peak_idx], 4), round(spectrum[peak_idx], 2)

    # Extracts top 2 peaks from a standard file: assumed to be compound and caffeine
    def detect_std_peaks(self, file_path: str) -> Optional[dict]:
        df = pd.read_csv(file_path)
        df = self.clean_column_names(df)
        time_col = df.columns[0]
        wavelengths = [float(c) for c in df.columns[1:] if c.replace('.', '', 1).isdigit()]
        closest_wl = self.find_closest_wavelength(wavelengths)
        closest_wl_col = str(closest_wl)

        spectrum = df[closest_wl_col].values
        peaks = self.extract_peaks(pd.Series(spectrum))
        if len(peaks) < self.min_peaks_required:
            return None

        # Get the top two peaks by intensity
        intensities = spectrum[peaks]
        top2_idx = [x for _, x in sorted(zip(intensities, peaks), reverse=True)[:2]]
        times = df[time_col].values
        return {
            "compound_time": round(times[top2_idx[0]], 4),
            "compound_intensity": round(spectrum[top2_idx[0]], 2),
            "caffeine_time": round(times[top2_idx[1]], 4),
            "caffeine_intensity": round(spectrum[top2_idx[1]], 2),
            "closest_wl": closest_wl_col,
            "time_col": time_col
        }

    # Processes both standard and reaction files for a given sample
    def process_file(self, file_path: str) -> Tuple[Optional[List], Optional[List]]:
        reaction_id = os.path.basename(file_path).split("_")[0]
        std_peaks = self.detect_std_peaks(file_path)
        if not std_peaks:
            print(f"{reaction_id}: Not enough peaks in STD file.")
            return None, None

        std_result = [
            reaction_id,
            std_peaks["compound_time"],
            std_peaks["compound_intensity"],
            std_peaks["caffeine_time"],
            std_peaks["caffeine_intensity"]
        ]

        rxn_file = os.path.join(self.data_folder, f"{reaction_id}_PDA.csv")
        if not os.path.exists(rxn_file):
            print(f"{reaction_id}: Reaction file missing.")
            return std_result, None

        try:
            df_rxn = pd.read_csv(rxn_file)
            df_rxn = self.clean_column_names(df_rxn)
            wl_col = std_peaks["closest_wl"]
            time_col = std_peaks["time_col"]

            if wl_col not in df_rxn.columns:
                print(f"{reaction_id}: Wavelength {wl_col} not found in reaction file.")
                return std_result, None

            # Try to find peaks in the reaction file near expected retention times
            comp_time, comp_int = self.get_peak_near(df_rxn, wl_col, time_col, std_peaks["compound_time"])
            caf_time, caf_int = self.get_peak_near(df_rxn, wl_col, time_col, std_peaks["caffeine_time"])

            rxn_result = [reaction_id, comp_time, comp_int, caf_time, caf_int]
            return std_result, rxn_result

        except Exception as e:
            print(f"{reaction_id}: Error in reaction file → {e}")
            return std_result, None

    # Calculates yield from standard and reaction intensities
    def calculate_yields(
        self,
        results_std: List[List],
        results_rxn: List[List],
        output_file: str = OutputFile.YIELD.value
    ) -> List[List]:
        try:
            df_std = pd.DataFrame(results_std, columns=[
                "reaction_id", "compound_retention_time", "compound_max_intensity",
                "caffeine_retention_time", "caffeine_max_intensity"])
            df_rxn = pd.DataFrame(results_rxn, columns=[
                "reaction_id", "compound_retention_time", "compound_max_intensity",
                "caffeine_retention_time", "caffeine_max_intensity"])

            df_merged = pd.merge(df_std, df_rxn, on="reaction_id", suffixes=('_std', '_rxn'))
            yield_results = []

            for _, row in df_merged.iterrows():
                try:
                    # Get max intensities for compound and caffeine in std and rxn
                    Ipmax, Icaffeinemax = row["compound_max_intensity_std"], row["caffeine_max_intensity_std"]
                    Ip, Icaffeine = row["compound_max_intensity_rxn"], row["caffeine_max_intensity_rxn"]
                    if any(pd.isna([Ipmax, Icaffeinemax, Ip, Icaffeine])) or Icaffeine == 0 or Icaffeinemax == 0:
                        raise ValueError("Invalid intensity data")

                    # Yield formula: (Ip/Icaffeine) / (Ipmax/Icaffeinemax)
                    y = (Ip / Icaffeine) / (Ipmax / Icaffeinemax)
                    yield_results.append([row["reaction_id"], f"{round(y * 100, 2):.2f}"])
                except Exception:
                    yield_results.append([row["reaction_id"], "0.00"])

            pd.DataFrame(yield_results, columns=["reaction_id", "yield (%)"]).to_csv(output_file, index=False)
            print(f"Yield results saved to '{output_file}'")
            return yield_results
        except Exception as e:
            print(f"Error calculating yields: {e}")
            return []

    # Main execution method to process all files in parallel
    def process_all(self):
        # Get all standard PDA files
        file_paths = glob.glob(os.path.join(self.data_folder, "*_std_PDA.csv"))
        with Pool() as pool:
            results = pool.map(self.process_file, file_paths)

        # Separate results for std and rxn
        results_std = [r[0] for r in results if r[0] is not None]
        results_rxn = [r[1] for r in results if r[1] is not None]

        # Save results to CSV
        pd.DataFrame(results_std, columns=[
            "reaction_id", "compound_retention_time", "compound_max_intensity",
            "caffeine_retention_time", "caffeine_max_intensity"]).to_csv(OutputFile.STD.value, index=False)
        print(f"Standard peak detection results saved to '{OutputFile.STD.value}'")

        pd.DataFrame(results_rxn, columns=[
            "reaction_id", "compound_retention_time", "compound_max_intensity",
            "caffeine_retention_time", "caffeine_max_intensity"]).to_csv(OutputFile.RXN.value, index=False)
        print(f"Reaction peak detection results saved to '{OutputFile.RXN.value}'")

        # Calculate yields
        self.calculate_yields(results_std, results_rxn)

if __name__ == "__main__":
    detector = PeakDetector()
    detector.process_all()
