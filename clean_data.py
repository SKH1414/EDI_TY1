import pandas as pd

def clean_data():
    input_file = "CSV/feature_vectors_syscallsbinders_frequency_5_Cat.csv"
    output_file = "cleaned_dataset.csv"
    
    print(f"Loading data from {input_file}...")
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Add a persistent ID column to trace samples back during obfuscation evaluation
    df.insert(0, 'apk_id', ['APK_' + str(i) for i in range(len(df))])
    
    print(f"Initial shape: {df.shape}")
    
    # Drop missing values if any
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs: {df.shape}")
    
    # Map classes to binary classification
    # Assuming CICMalDroid2020 categories:
    # 5: Benign -> 0
    # 1-4: Malware (Adware, Banking, SMS, Riskware) -> 1
    
    if 'Class' in df.columns:
        print("Mapping 'Class' column to binary labels...")
        df['Label'] = df['Class'].apply(lambda x: 0 if x == 5 else 1)
        # Drop the original 'Class' column
        df.drop('Class', axis=1, inplace=True)
        
        print("Class distribution after mapping:")
        print(df['Label'].value_counts())
    else:
        print("Warning: 'Class' column not found.")
        
    print(f"Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    clean_data()
