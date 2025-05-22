# LCMS Yield Estimation Using Product Standards

This repository contains a Python-based workflow for estimating chemical reaction yields from LCMS/UV data using a product standard and an internal reference (caffeine). 
The workflow was developed as part of a research task in the context of high-throughput reaction analysis and cheminformatics-driven yield prediction.

In high-throughput experimental (HTE) chemistry, accurately estimating the yield of reactions is essential for reaction optimization and machine learning model training. 
When a purified product standard is available, UV signal intensities can be normalized and used to compute yields reliably, even when absolute quantification is difficult due to instrument variability.

This pipeline addresses yield computation by:
- Integrating UV/Vis chromatograms at 254 nm.
- Normalizing product intensities using a caffeine internal standard.
- Estimate yield using reaction and product standard measurements.
- Automating the entire process for batch analysis.

## Project Structure

# Main script for yield estimation
- estimate_yields.py
- yields.csv
- find_MS_peaks.py


📂 data/ # LCMS/UV data files for each reaction and molecular_weights.json file with the molecular weights of expected products
- molecular_weights.json # Molecular weights of expected products
- R5985_MS_minus.csv
- R5985_MS_plus.csv
- R5985_PDA.csv

## ⚙️ estimate_yields.py

The script estimates yields by:
1. Parsing LCMS/UV data from multiple reaction and product runs.
2. Identifying retention times and extracting intensity peaks.
3. Normalizing product intensity by caffeine signal.
4. Calculating the yield as a ratio of normalized intensities (reaction/product standard).

`Yield = (I_product_reaction / I_caffeine_reaction) / (I_product_standard / I_caffeine_standard)`

