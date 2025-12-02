"""
Simple debug tool for data loading issues - No emojis
"""
import pandas as pd
from pathlib import Path

def debug_data_loading():
    print("DEBUG: Data Loading Issues")
    print("=" * 50)
    
    # Check if files exist
    files_to_check = [
        'data/raw/titanic.csv',
        'data/processed/titanic_processed.csv',
        'data/processed/X_train_scaled.csv', 
        'data/processed/X_test_scaled.csv',
        'data/processed/y_train.csv',
        'data/processed/y_test.csv'
    ]
    
    print("Checking file existence:")
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"   OK: {file_path}")
            try:
                # Try to read the file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(path)
                    print(f"      Shape: {df.shape}, Columns: {len(df.columns)}")
                else:
                    print(f"      (Not a CSV file)")
            except Exception as e:
                print(f"      READ ERROR: {e}")
        else:
            print(f"   MISSING: {file_path}")
    
    print("\nChecking data/raw/titanic.csv structure:")
    raw_path = Path('data/raw/titanic.csv')
    if raw_path.exists():
        try:
            df_raw = pd.read_csv(raw_path)
            print(f"   Successfully loaded raw data")
            print(f"   Shape: {df_raw.shape}")
            print(f"   Columns: {list(df_raw.columns)}")
            print(f"   First few rows:")
            print(df_raw.head(3))
            
            # Check for required columns
            required_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            if missing_cols:
                print(f"   MISSING COLUMNS: {missing_cols}")
            else:
                print(f"   All required columns present")
                
        except Exception as e:
            print(f"   Error reading file: {e}")
    
    print("\nRecommended fix:")
    print("   1. Delete the data/processed/ folder")
    print("   2. Run: python download_data.py")
    print("   3. Run: python preprocess_data.py")
    print("   4. Run: python train_model.py")

if __name__ == "__main__":
    debug_data_loading()