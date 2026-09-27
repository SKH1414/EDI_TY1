import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from config import SEED

def main():
    print("Loading dataset...")
    df = pd.read_parquet('cleaned_dataset.parquet')
    
    # Stratified split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.3, random_state=SEED, stratify=df['label']
    )
    
    # 50/50 split of the 30% temp gives 15% val, 15% test
    # Get labels for temp split to stratify again
    temp_labels = df.loc[temp_idx, 'label']
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=temp_labels
    )
    
    split_info = {
        'seed': SEED,
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist()
    }
    
    with open('split_indices.json', 'w') as f:
        json.dump(split_info, f)
        
    print(f"Saved splits to split_indices.json: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test.")

if __name__ == "__main__":
    main()
