"""
STEP 4: Data Preprocessing and Feature Engineering
Prepare the data for machine learning modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Titanic dataset
    Handles missing values, feature engineering, and data preparation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputation_values_ = {}
        self.feature_names_ = []
        
    def load_data(self):
        """Load the raw Titanic dataset"""
        data_path = Path('data/raw/titanic.csv')
        self.df = pd.read_csv(data_path)
        print(f"DATA: Loaded {len(self.df)} passenger records")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("PREPROCESSING: Handling missing values...")
        
        # Store imputation values for consistency
        self.imputation_values_['Age'] = self.df['Age'].median()
        self.imputation_values_['Fare'] = self.df['Fare'].median()
        self.imputation_values_['Embarked'] = self.df['Embarked'].mode()[0]
        
        # Apply imputation
        self.df['Age'] = self.df['Age'].fillna(self.imputation_values_['Age'])
        self.df['Fare'] = self.df['Fare'].fillna(self.imputation_values_['Fare'])
        self.df['Embarked'] = self.df['Embarked'].fillna(self.imputation_values_['Embarked'])
        
        # Drop Cabin due to high missingness (we'll extract deck info first)
        if 'Cabin' in self.df.columns:
            # Extract deck from cabin before dropping
            self.df['Deck'] = self.df['Cabin'].str[0]
            self.df['Deck'] = self.df['Deck'].fillna('Unknown')
            self.df = self.df.drop('Cabin', axis=1)
        
        print("COMPLETED: Missing values handled")
    
    def extract_titles(self):
        """Extract and categorize titles from passenger names"""
        print("FEATURE ENGINEERING: Extracting titles from names...")
        
        # Extract title from name
        self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Consolidate titles
        title_mapping = {
            'Mlle': 'Miss', 
            'Ms': 'Miss', 
            'Mme': 'Mrs',
            'Col': 'Officer', 
            'Major': 'Officer', 
            'Dr': 'Officer',
            'Rev': 'Officer', 
            'Jonkheer': 'Royalty', 
            'Don': 'Royalty',
            'Dona': 'Royalty', 
            'Countess': 'Royalty', 
            'Sir': 'Royalty', 
            'Lady': 'Royalty', 
            'Capt': 'Officer'
        }
        
        self.df['Title'] = self.df['Title'].replace(title_mapping)
        
        # Group rare titles
        title_counts = self.df['Title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        self.df['Title'] = self.df['Title'].replace(rare_titles, 'Rare')
        
        print(f"CREATED: Title feature with {self.df['Title'].nunique()} categories")
    
    def create_family_features(self):
        """Create family-related features"""
        print("FEATURE ENGINEERING: Creating family features...")
        
        # Family size
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        
        # Is alone
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Family categories
        self.df['FamilyCategory'] = pd.cut(
            self.df['FamilySize'], 
            bins=[0, 1, 4, 11], 
            labels=['Alone', 'Small', 'Large']
        )
        
        print("CREATED: Family size, IsAlone, and FamilyCategory features")
    
    def create_age_groups(self):
        """Create age group categories"""
        print("FEATURE ENGINEERING: Creating age groups...")
        
        self.df['AgeGroup'] = pd.cut(
            self.df['Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
        )
        
        print("CREATED: AgeGroup feature")
    
    def create_fare_groups(self):
        """Create fare categories"""
        print("FEATURE ENGINEERING: Creating fare groups...")
        
        self.df['FareGroup'] = pd.qcut(self.df['Fare'], 4, 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
        
        print("CREATED: FareGroup feature")
    
    def encode_categorical_features(self):
        """Encode categorical variables for modeling"""
        print("ENCODING: Converting categorical features...")
        
        # List of categorical columns to encode
        categorical_columns = [
            'Sex', 'Embarked', 'Title', 'Pclass', 
            'AgeGroup', 'FareGroup', 'Deck', 'FamilyCategory'
        ]
        
        # Only encode columns that exist in the dataframe
        existing_categorical = [col for col in categorical_columns if col in self.df.columns]
        
        # Create dummy variables
        self.df = pd.get_dummies(self.df, columns=existing_categorical, drop_first=True)
        
        print(f"ENCODED: {len(existing_categorical)} categorical features")
    
    def select_final_features(self):
        """Select final features for modeling and drop unnecessary columns"""
        print("FEATURE SELECTION: Preparing final feature set...")
        
        # Columns to always drop
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        
        # Only drop columns that exist
        existing_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_to_drop)
        
        # Ensure Survived is the first column if it exists
        if 'Survived' in self.df.columns:
            cols = ['Survived'] + [col for col in self.df.columns if col != 'Survived']
            self.df = self.df[cols]
        
        self.feature_names_ = [col for col in self.df.columns if col != 'Survived']
        
        print(f"FINAL FEATURES: {len(self.feature_names_)} features selected")
        print(f"FEATURE NAMES: {self.feature_names_}")
    
    def split_data(self):
        """Split data into training and testing sets"""
        print("DATA SPLITTING: Creating train-test split...")
        
        # Separate features and target
        X = self.df.drop('Survived', axis=1)
        y = self.df['Survived']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y  # Important for imbalanced datasets
        )
        
        print(f"TRAINING SET: {len(self.X_train)} samples")
        print(f"TESTING SET: {len(self.X_test)} samples")
        print(f"FEATURES: {self.X_train.shape[1]} features")
    
    def scale_features(self):
        """Scale numerical features"""
        print("SCALING: Standardizing numerical features...")
        
        # Identify numerical columns (excluding binary encoded ones)
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        # Scale features
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.X_train_scaled[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test_scaled[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        print("COMPLETED: Feature scaling")
    
    def save_processed_data(self):
        """Save processed data and preprocessing objects"""
        print("SAVING: Saving processed data...")
        
        # Create processed data directory
        processed_dir = Path('data/processed')
        processed_dir.mkdir(exist_ok=True)
        
        # Save processed dataset
        self.df.to_csv(processed_dir / 'titanic_processed.csv', index=False)
        
        # Save train/test splits
        self.X_train.to_csv(processed_dir / 'X_train.csv', index=False)
        self.X_test.to_csv(processed_dir / 'X_test.csv', index=False)
        self.y_train.to_csv(processed_dir / 'y_train.csv', index=False)
        self.y_test.to_csv(processed_dir / 'y_test.csv', index=False)
        
        # Save scaled versions
        self.X_train_scaled.to_csv(processed_dir / 'X_train_scaled.csv', index=False)
        self.X_test_scaled.to_csv(processed_dir / 'X_test_scaled.csv', index=False)
        
        # Save preprocessing objects
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.scaler, models_dir / 'scaler.pkl')
        joblib.dump(self.imputation_values_, models_dir / 'imputation_values.pkl')
        
        print("SAVED: All processed data and preprocessing objects")
    
    def generate_preprocessing_report(self):
        """Generate a report on the preprocessing steps"""
        print("REPORT: Generating preprocessing report...")
        
        report = f"""# Data Preprocessing Report

## Dataset Overview
- **Original Samples**: {len(self.df)}
- **Final Features**: {len(self.feature_names_)}
- **Training Samples**: {len(self.X_train)}
- **Testing Samples**: {len(self.X_test)}

## Preprocessing Steps Applied

### 1. Missing Value Handling
- **Age**: Imputed with median ({self.imputation_values_['Age']:.1f} years)
- **Fare**: Imputed with median (${self.imputation_values_['Fare']:.2f})
- **Embarked**: Imputed with mode ({self.imputation_values_['Embarked']})
- **Cabin**: Extracted deck information, then dropped due to high missingness

### 2. Feature Engineering
- **Title Extraction**: Created from passenger names
- **Family Features**: FamilySize, IsAlone, FamilyCategory
- **Age Groups**: Child, Teen, Young Adult, Adult, Senior
- **Fare Groups**: Low, Medium, High, Very High

### 3. Encoding
- All categorical variables converted to dummy variables
- First category dropped to avoid multicollinearity

### 4. Feature Scaling
- Numerical features standardized using StandardScaler
- Mean = 0, Standard Deviation = 1

## Final Feature Set
{', '.join(self.feature_names_)}

## Data Splits
- **Training Set**: {len(self.X_train)} samples (80%)
- **Test Set**: {len(self.X_test)} samples (20%)
- **Stratified**: Yes (maintains survival rate distribution)

## Next Steps
1. Proceed to model training with prepared datasets
2. Use scaled features for algorithms sensitive to feature scales
3. Use original features for tree-based models
"""
        
        report_path = Path('docs/preprocessing_report.md')
        report_path.write_text(report)
        print(f"CREATED: {report_path}")
    
    def run_complete_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("PHASE 4: Starting Data Preprocessing Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Preprocessing steps
        self.handle_missing_values()
        self.extract_titles()
        self.create_family_features()
        self.create_age_groups()
        self.create_fare_groups()
        self.encode_categorical_features()
        self.select_final_features()
        
        # Data preparation
        self.split_data()
        self.scale_features()
        
        # Save results
        self.save_processed_data()
        self.generate_preprocessing_report()
        
        print("\nPHASE 4 COMPLETE!")
        print("\nCREATED OUTPUTS:")
        print("  - Processed dataset: data/processed/titanic_processed.csv")
        print("  - Train/test splits: data/processed/X_train.csv, X_test.csv, etc.")
        print("  - Scaled features: data/processed/X_train_scaled.csv, etc.")
        print("  - Preprocessing objects: models/scaler.pkl, models/imputation_values.pkl")
        print("  - Preprocessing report: docs/preprocessing_report.md")
        print("\nNEXT STEPS:")
        print("  1. Review: docs/preprocessing_report.md")
        print("  2. Run: python train_model.py")

def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    preprocessor.run_complete_preprocessing()

if __name__ == "__main__":
    main()