# Data Preprocessing Report

## Dataset Overview
- **Original Samples**: 891
- **Final Features**: 33
- **Training Samples**: 712
- **Testing Samples**: 179

## Preprocessing Steps Applied

### 1. Missing Value Handling
- **Age**: Imputed with median (28.0 years)
- **Fare**: Imputed with median ($14.45)
- **Embarked**: Imputed with mode (S)
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
Age, SibSp, Parch, Fare, FamilySize, IsAlone, Sex_male, Embarked_Q, Embarked_S, Title_Miss, Title_Mr, Title_Mrs, Title_Officer, Title_Rare, Pclass_2, Pclass_3, AgeGroup_Teen, AgeGroup_Young Adult, AgeGroup_Adult, AgeGroup_Senior, FareGroup_Medium, FareGroup_High, FareGroup_Very High, Deck_B, Deck_C, Deck_D, Deck_E, Deck_F, Deck_G, Deck_T, Deck_Unknown, FamilyCategory_Small, FamilyCategory_Large

## Data Splits
- **Training Set**: 712 samples (80%)
- **Test Set**: 179 samples (20%)
- **Stratified**: Yes (maintains survival rate distribution)

## Next Steps
1. Proceed to model training with prepared datasets
2. Use scaled features for algorithms sensitive to feature scales
3. Use original features for tree-based models
