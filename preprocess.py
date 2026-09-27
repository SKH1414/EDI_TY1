import os
import csv
import json
import pandas as pd
import numpy as np

def preprocess_cicmaldroid(
    static_csv_path,
    dynamic_csv_path,
    output_parquet_path,
    summary_json_path,
    min_nonzero_count=10
):
    print(f"Starting preprocessing of CICMalDroid2020 dataset.")
    temp_utf8_csv = static_csv_path + ".utf8.tmp"
    header_txt_path = static_csv_path + ".raw_header.txt"
    
    # 1. Convert UTF-7 to UTF-8 and extract header
    if not os.path.exists(temp_utf8_csv):
        print("Converting UTF-7 CSV to UTF-8 without header...")
        with open(static_csv_path, 'r', encoding='utf-7', errors='replace') as fin:
            reader = csv.reader(fin)
            with open(temp_utf8_csv, 'w', encoding='utf-8', newline='') as fout:
                writer = csv.writer(fout)
                try:
                    header = next(reader)
                    with open(header_txt_path, 'w', encoding='utf-8') as hf:
                        hf.write(",".join(header) + "\n")
                    print(f"Saved raw header to {header_txt_path}")
                except StopIteration:
                    pass
                
                # Write rest of the data
                for row in reader:
                    writer.writerow(row)
        print("UTF-8 conversion complete.")
    else:
        print(f"Using existing UTF-8 temp file: {temp_utf8_csv}")

    # 2 & 3. Pass 1: Find non-numeric columns
    print("Scanning for non-numeric columns in chunks...")
    non_numeric_cols = set()
    total_cols = None
    
    # Read chunk by chunk to find non-numeric columns
    chunk_iter = pd.read_csv(temp_utf8_csv, header=None, chunksize=2000, low_memory=False, dtype=str)
    for i, chunk in enumerate(chunk_iter):
        if total_cols is None:
            total_cols = chunk.shape[1]
        
        # Drop column 0
        if 0 in chunk.columns:
            chunk = chunk.drop(columns=[0])
            
        for col in chunk.columns:
            if col in non_numeric_cols:
                continue
            
            # Mask of non-empty strings
            non_empty = chunk[col].fillna('').str.strip() != ''
            non_empty_series = chunk[col][non_empty]
            
            if len(non_empty_series) > 0:
                converted = pd.to_numeric(non_empty_series, errors='coerce')
                if converted.isna().any():
                    # Found a non-numeric value
                    non_numeric_cols.add(col)
                    sample_val = non_empty_series[converted.isna()].iloc[0]
                    print(f"Column {col} is non-numeric (sample: '{sample_val}')")
                    
    print(f"Found {len(non_numeric_cols)} non-numeric columns.")

    # 4. Pass 2: Load data, convert to numeric (uint16/uint32), keep in memory
    print("Loading data, dropping non-numeric columns, and converting to numeric...")
    numeric_chunks = []
    chunk_iter = pd.read_csv(temp_utf8_csv, header=None, chunksize=2000, low_memory=False, dtype=str)
    
    for i, chunk in enumerate(chunk_iter):
        # Drop column 0 and non-numeric columns
        cols_to_drop = [0] + list(non_numeric_cols)
        chunk = chunk.drop(columns=[c for c in cols_to_drop if c in chunk.columns])
        
        # Rename columns to positional names
        # Original columns were 0 to total_cols-1.
        # Let's map original column index to feat_XXXXX
        col_mapping = {c: f"feat_{c:05d}" for c in chunk.columns}
        chunk = chunk.rename(columns=col_mapping)
        
        # Replace empty strings with '0', then convert to float, then uint32
        chunk = chunk.replace('', '0').fillna('0')
        chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.uint32)
        numeric_chunks.append(chunk_num)
        print(f"Processed chunk {i+1}")

    print("Concatenating chunks...")
    full_matrix = pd.concat(numeric_chunks, ignore_index=True)
    print(f"Full matrix shape: {full_matrix.shape}")
    
    # 5. Load dynamic labels
    print("Loading dynamic features file for labels...")
    dynamic_df = pd.read_csv(dynamic_csv_path, usecols=['Class'])
    assert len(dynamic_df) == len(full_matrix), f"Row count mismatch! Static: {len(full_matrix)}, Dynamic: {len(dynamic_df)}"
    
    # Class mapping: 5 = Benign (0), else = Malware (1)
    full_matrix['label'] = (dynamic_df['Class'] != 5).astype(np.uint8)
    
    benign_count = (full_matrix['label'] == 0).sum()
    malware_count = (full_matrix['label'] == 1).sum()
    print(f"Label balance - Benign: {benign_count}, Malware: {malware_count}")

    # 6 & 7. Feature selection
    print("Computing column statistics and applying feature selection...")
    feature_cols = [c for c in full_matrix.columns if c != 'label']
    
    # Compute nonzero counts
    nonzero_counts = (full_matrix[feature_cols] > 0).sum(axis=0)
    
    # Identify constant and rare columns
    constant_cols = nonzero_counts[(nonzero_counts == 0) | (nonzero_counts == len(full_matrix))].index.tolist()
    rare_cols = nonzero_counts[(nonzero_counts > 0) & (nonzero_counts < min_nonzero_count)].index.tolist()
    
    # Some columns might be in both if min_nonzero_count > 0, but the conditions are mutually exclusive if min_nonzero_count > 1 and len > 1
    cols_to_drop_fs = set(constant_cols + rare_cols)
    
    print(f"Dropping {len(constant_cols)} constant columns.")
    print(f"Dropping {len(rare_cols)} rare columns (nonzero in < {min_nonzero_count} samples).")
    
    final_cols = [c for c in full_matrix.columns if c not in cols_to_drop_fs]
    final_matrix = full_matrix[final_cols]
    
    print(f"Final matrix shape: {final_matrix.shape}")

    # 8. Save output
    print(f"Saving to {output_parquet_path}...")
    final_matrix.to_parquet(output_parquet_path, index=False)
    
    # Summary
    summary = {
        "original_total_cols": total_cols,
        "dropped_col_0": True,
        "dropped_non_numeric_cols_count": len(non_numeric_cols),
        "dropped_non_numeric_cols": list(non_numeric_cols),
        "dropped_constant_cols_count": len(constant_cols),
        "dropped_rare_cols_count": len(rare_cols),
        "min_nonzero_count_threshold": min_nonzero_count,
        "final_feature_count": final_matrix.shape[1] - 1, # excluding label
        "final_samples": final_matrix.shape[0],
        "class_balance": {
            "benign_0": int(benign_count),
            "malware_1": int(malware_count)
        },
        "feature_naming_convention": "feat_XXXXX (where XXXXX is the original 0-indexed column position, including column 0 which was dropped)",
        "header_reference_file": header_txt_path
    }
    
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved summary to {summary_json_path}")
    
    # 9. Final summary print
    mem_usage_mb = final_matrix.memory_usage(deep=True).sum() / (1024**2)
    print("\n--- Final Summary ---")
    print(f"Final matrix shape: {final_matrix.shape}")
    print(f"Class balance: {benign_count} Benign (0) / {malware_count} Malware (1)")
    print(f"Memory footprint: {mem_usage_mb:.2f} MB")
    print(f"Cleaned dataset saved to: {output_parquet_path}")
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Raw header saved to: {header_txt_path}")

if __name__ == "__main__":
    preprocess_cicmaldroid(
        static_csv_path="feature_vectors_static.csv",
        dynamic_csv_path="feature_vectors_syscallsbinders_frequency_5_Cat.csv",
        output_parquet_path="cleaned_dataset.parquet",
        summary_json_path="preprocessing_summary.json",
        min_nonzero_count=10
    )
