"""
STEP 2: Download and explore the Titanic dataset
Without emojis to avoid encoding issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import ssl

def download_titanic_data():
    """Download the Titanic dataset from reliable source"""
    
    print("STEP 2: Downloading Titanic Dataset...")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (we'll try multiple sources)
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    ]
    
    for i, url in enumerate(urls):
        try:
            print(f"Attempting download from source {i+1}...")
            
            # Bypass SSL verification for download
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Download the file
            df = pd.read_csv(url)
            
            # Save to raw data directory
            output_path = Path('data/raw/titanic.csv')
            df.to_csv(output_path, index=False)
            
            print(f"SUCCESS: Dataset downloaded!")
            print(f"  Shape: {df.shape}")
            print(f"  Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"FAILED: Source {i+1} failed: {e}")
            continue
    
    print("ERROR: All download attempts failed. Please check your internet connection.")
    return None

def explore_dataset(df):
    """Do initial exploration of the dataset"""
    
    print("\nINITIAL DATA EXPLORATION:")
    print("=" * 30)
    
    # Basic information
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Passengers: {len(df)}")
    print(f"Number of Features: {len(df.columns)}")
    
    # Column information
    print("\nCOLUMNS AND DATA TYPES:")
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        print(f"  - {col}: {dtype} ({missing} missing)")
    
    # Survival overview
    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean() * 100
        survived_count = df['Survived'].sum()
        print(f"\nSURVIVAL OVERVIEW:")
        print(f"  Survival Rate: {survival_rate:.1f}%")
        print(f"  Survived: {survived_count} passengers")
        print(f"  Did Not Survive: {len(df) - survived_count} passengers")
    
    # Basic statistics
    print("\nBASIC STATISTICS:")
    print(f"  Youngest Passenger: {df['Age'].min():.1f} years")
    print(f"  Oldest Passenger: {df['Age'].max():.1f} years")
    print(f"  Average Age: {df['Age'].mean():.1f} years")
    print(f"  Minimum Fare: ${df['Fare'].min():.2f}")
    print(f"  Maximum Fare: ${df['Fare'].max():.2f}")
    print(f"  Average Fare: ${df['Fare'].mean():.2f}")
    
    return df

def create_data_summary(df):
    """Create a summary markdown file about the dataset"""
    
    summary = f"""# Titanic Dataset Summary

## Basic Information
- **Total Passengers**: {len(df)}
- **Number of Features**: {len(df.columns)}
- **Dataset Source**: Multiple historical records

## Target Variable
- **Survival Rate**: {df['Survived'].mean()*100:.1f}%
- **Survived**: {df['Survived'].sum()} passengers
- **Did Not Survive**: {len(df) - df['Survived'].sum()} passengers

## Features Overview

### Numerical Features
- **Age**: {df['Age'].min():.1f} to {df['Age'].max():.1f} years (mean: {df['Age'].mean():.1f})
- **Fare**: ${df['Fare'].min():.2f} to ${df['Fare'].max():.2f} (mean: ${df['Fare'].mean():.2f})
- **Siblings/Spouses**: 0 to {df['SibSp'].max()}
- **Parents/Children**: 0 to {df['Parch'].max()}

### Categorical Features
- **Passenger Classes**: 1st, 2nd, 3rd
- **Gender**: Male, Female
- **Embarkation Ports**: {', '.join([str(x) for x in df['Embarked'].unique() if pd.notna(x)])}

## Data Quality Notes
- **Missing Age Values**: {df['Age'].isnull().sum()} ({df['Age'].isnull().sum()/len(df)*100:.1f}%)
- **Missing Cabin Values**: {df['Cabin'].isnull().sum()} ({df['Cabin'].isnull().sum()/len(df)*100:.1f}%)
- **Missing Embarked Values**: {df['Embarked'].isnull().sum()}

## Next Steps
1. Handle missing values
2. Explore feature relationships with survival
3. Create visualizations
4. Build predictive models
"""
    
    Path('docs/data_summary.md').write_text(summary)
    print("CREATED: docs/data_summary.md")

def main():
    """Main data download function"""
    df = download_titanic_data()
    
    if df is not None:
        explore_dataset(df)
        create_data_summary(df)
        
        print("\nPHASE 2 COMPLETE!")
        print("\nNEXT STEPS:")
        print("   1. Review: docs/data_summary.md")
        print("   2. Run: python explore_data.py")
        print("   3. Start building visualizations!")
    else:
        print("\nERROR: Data download failed. Please check your connection and try again.")

if __name__ == "__main__":
    main()