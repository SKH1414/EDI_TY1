# CICMalDroid2020: Dataset & Preprocessing Summary

This document explains the origins of the data, the exact preprocessing pipeline applied to clean it, and the final artifacts produced for downstream machine learning.

## 1. The Dataset
The project utilizes the **CICMalDroid2020** dataset, specifically targeting the **static features** extracted from Android APK manifests for binary classification (Benign vs. Malware).

**Raw Dataset Characteristics:**
- **Samples**: 11,598 APK files.
- **Features**: 50,621 raw static features (permissions, intents, API calls).
- **Encoding**: UTF-7 (with several malformed shift sequences that historically broke standard CSV parsers).
- **Headers**: The header row was heavily malformed due to unescaped quote characters merging adjacent strings, rendering direct column-to-name mapping unreliable past the first corruption point.
- **Labels**: The static dataset did not contain target labels natively. Labels had to be joined positionally from the companion dynamic dataset (`feature_vectors_syscallsbinders_frequency_5_Cat.csv`), which categorized the apps into 5 classes (1=Adware, 2=Banking, 3=SMS malware, 4=Riskware, 5=Benign).

## 2. Preprocessing Steps
To prepare the dataset for binary classification, a highly optimized, chunked pipeline was executed (`preprocess.py`):

1. **Format Correction & Header Extraction**:
   - Safely streamed the file from UTF-7 to UTF-8 using `errors='replace'` to bypass malformed bytes.
   - Extracted the broken header line into a separate reference text file and removed it from the active pipeline to avoid parsing crashes.

2. **Non-Numeric Pruning**:
   - The matrix was loaded in chunks to avoid memory overflow.
   - Scanned all 50,620 features to locate and drop **15 non-numeric columns** (which contained per-app identity strings like package names, e.g., `'com.droid.snail'`, or literal string booleans like `'True'`).

3. **Memory Optimization**:
   - Replaced empty string cells with `0`.
   - Downcast all features to 32-bit unsigned integers (`uint32`), shrinking the potential ~4.7GB float matrix down to a highly manageable ~136 MB.

4. **Label Integration**:
   - Merged the "Class" column from the dynamic dataset using row position.
   - Mapped to a binary target constraint: **Benign (Class 5) -> `0`**, **Malware (Classes 1-4) -> `1`**.

5. **Aggressive Feature Selection**:
   - Dropped **18,953 constant columns** (features that were present in either 0 or all 11,598 samples).
   - Dropped **28,574 rare columns** (features present in fewer than 10 samples, minimizing noise and preventing overfitting on ultra-rare parameters).
   - Transformed the raw 50,620 feature space into a dense, clean space of **3,078 features**.

## 3. Artifacts Obtained
Following the execution of `preprocess.py`, the following artifacts were generated and persisted to the workspace:

- **`cleaned_dataset.parquet`**: The final, ML-ready feature matrix containing 11,598 rows and 3,079 columns (3,078 positional features `feat_00000...` + 1 `label` column). The class balance is exactly 1,795 Benign and 9,803 Malware. Saved as Parquet for rapid I/O.
- **`preprocessing_summary.json`**: A metadata JSON tracking exactly what was dropped (including the names of the 15 string columns), the filtering thresholds used, and the final shape balances for audit trails.
- **`feature_vectors_static.csv.raw_header.txt`**: The original UTF-8 parsed header string. While the feature columns are intentionally positionally named (e.g., `feat_00014`) to bypass parser corruption, this file allows researchers to manually count indices to map a specific feature back to its human-readable API/Permission string if it proves highly important during model evaluation.
- **`preprocess.py`**: The reproducible Python script built to execute this entire pipeline in chunks automatically.
