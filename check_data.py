"""
Check if all required files exist
"""
from pathlib import Path

def check_files():
    print("Checking Required Files for Streamlit App...")
    print("=" * 50)
    
    required_files = [
        'data/raw/titanic.csv',
        'data/processed/titanic_processed.csv',
        'data/processed/X_train_scaled.csv',
        'data/processed/X_test_scaled.csv',
        'data/processed/y_train.csv', 
        'data/processed/y_test.csv',
        'models/best_model.pkl',
        'models/scaler.pkl',
        'models/imputation_values.pkl'
    ]
    
    all_good = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"OK: {file_path}")
        else:
            print(f"MISSING: {file_path}")
            all_good = False
    
    return all_good

def main():
    if check_files():
        print("\nSUCCESS: All files present!")
        print("Run: python -m streamlit run app_final.py")
    else:
        print("\nMISSING FILES: Please run:")
        print("python preprocess_data.py")
        print("python train_model.py")

if __name__ == "__main__":
    main()