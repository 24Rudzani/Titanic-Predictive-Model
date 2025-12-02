"""
STEP 3: Comprehensive Data Exploration and Visualization - FIXED VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class DataExplorer:
    """Comprehensive data exploration and visualization"""
    
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the Titanic dataset"""
        data_path = Path('data/raw/titanic.csv')
        self.df = pd.read_csv(data_path)
        print("DATA: Loaded Titanic dataset")
        return self.df
    
    def create_demographic_analysis(self):
        """Analyze demographic factors and survival"""
        print("ANALYSIS: Creating demographic analysis...")
        
        # Create figures directory
        fig_dir = Path('presentation/assets')
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Survival by Passenger Class
        plt.figure(figsize=(10, 6))
        survival_by_class = pd.crosstab(self.df['Pclass'], self.df['Survived'])
        survival_by_class.plot(kind='bar', stacked=True)
        plt.title('Survival by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Number of Passengers')
        plt.legend(['Did Not Survive', 'Survived'])
        plt.tight_layout()
        plt.savefig(fig_dir / 'survival_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Survival by Gender
        plt.figure(figsize=(10, 6))
        survival_by_gender = pd.crosstab(self.df['Sex'], self.df['Survived'])
        survival_by_gender.plot(kind='bar', stacked=True)
        plt.title('Survival by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Number of Passengers')
        plt.legend(['Did Not Survive', 'Survived'])
        plt.tight_layout()
        plt.savefig(fig_dir / 'survival_by_gender.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Age distribution by survival
        plt.figure(figsize=(12, 6))
        
        # Remove missing ages for this plot
        age_data = self.df[self.df['Age'].notna()]
        
        plt.hist([age_data[age_data['Survived'] == 0]['Age'], 
                  age_data[age_data['Survived'] == 1]['Age']], 
                 bins=20, alpha=0.7, label=['Did Not Survive', 'Survived'])
        plt.title('Age Distribution by Survival Status')
        plt.xlabel('Age')
        plt.ylabel('Number of Passengers')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Demographic analysis plots")
    
    def create_fare_analysis(self):
        """Analyze fare patterns and survival"""
        print("ANALYSIS: Creating fare analysis...")
        
        fig_dir = Path('presentation/assets')
        
        # Fare distribution by class and survival
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=self.df)
        plt.title('Fare Distribution by Class and Survival')
        plt.tight_layout()
        plt.savefig(fig_dir / 'fare_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Fare analysis plot")
    
    def create_family_analysis(self):
        """Analyze family size impact on survival"""
        print("ANALYSIS: Creating family analysis...")
        
        # Create family size feature
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        fig_dir = Path('presentation/assets')
        
        # Family size vs survival
        plt.figure(figsize=(12, 6))
        family_survival = self.df.groupby('FamilySize')['Survived'].mean()
        plt.plot(family_survival.index, family_survival.values, marker='o', linewidth=2)
        plt.title('Survival Rate by Family Size')
        plt.xlabel('Family Size')
        plt.ylabel('Survival Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'family_size_survival.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Alone vs with family
        plt.figure(figsize=(8, 6))
        alone_survival = self.df.groupby('IsAlone')['Survived'].mean()
        alone_survival.plot(kind='bar')
        plt.title('Survival Rate: Alone vs With Family')
        plt.xlabel('Is Alone (0=With Family, 1=Alone)')
        plt.ylabel('Survival Rate')
        plt.tight_layout()
        plt.savefig(fig_dir / 'alone_vs_family.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Family analysis plots")
    
    def create_correlation_analysis(self):
        """Create correlation matrix and heatmap"""
        print("ANALYSIS: Creating correlation analysis...")
        
        # Select numerical columns for correlation
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr()
        
        fig_dir = Path('presentation/assets')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(fig_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Correlation matrix")
        
        # Print top correlations with survival
        survival_corr = correlation_matrix['Survived'].sort_values(ascending=False)
        print("\nTOP CORRELATIONS WITH SURVIVAL:")
        for feature, corr in survival_corr.items():
            if feature != 'Survived':
                print(f"  {feature}: {corr:+.3f}")
    
    def create_embarkation_analysis(self):
        """Analyze embarkation port patterns"""
        print("ANALYSIS: Creating embarkation analysis...")
        
        fig_dir = Path('presentation/assets')
        
        # Survival by embarkation port
        plt.figure(figsize=(10, 6))
        embark_survival = pd.crosstab(self.df['Embarked'], self.df['Survived'])
        embark_survival.plot(kind='bar', stacked=True)
        plt.title('Survival by Embarkation Port')
        plt.xlabel('Embarkation Port')
        plt.ylabel('Number of Passengers')
        plt.legend(['Did Not Survive', 'Survived'])
        plt.tight_layout()
        plt.savefig(fig_dir / 'embarkation_survival.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Embarkation analysis plot")
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("REPORT: Generating insights report...")
        
        # Calculate key metrics
        overall_survival = self.df['Survived'].mean()
        
        # Class insights
        class_survival = self.df.groupby('Pclass')['Survived'].mean()
        class_1_survival = class_survival[1]
        class_3_survival = class_survival[3]
        
        # Gender insights
        gender_survival = self.df.groupby('Sex')['Survived'].mean()
        female_survival = gender_survival['female']
        male_survival = gender_survival['male']
        
        # Age insights
        child_survival = self.df[self.df['Age'] < 18]['Survived'].mean()
        adult_survival = self.df[(self.df['Age'] >= 18) & (self.df['Age'] < 60)]['Survived'].mean()
        senior_survival = self.df[self.df['Age'] >= 60]['Survived'].mean()
        
        # Handle NaN for senior survival
        senior_survival_text = f"{senior_survival:.1%}" if not np.isnan(senior_survival) else 'N/A'
        
        # Family insights
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        small_family_survival = self.df[self.df['FamilySize'].between(2, 4)]['Survived'].mean()
        alone_survival = self.df[self.df['FamilySize'] == 1]['Survived'].mean()
        
        insights = f"""# Titanic Data Insights Report

## Key Survival Statistics
- **Overall Survival Rate**: {overall_survival:.1%}
- **1st Class Survival**: {class_1_survival:.1%} ({(class_1_survival/overall_survival - 1):.0%} above average)
- **3rd Class Survival**: {class_3_survival:.1%} ({(class_3_survival/overall_survival - 1):.0%} below average)
- **Female Survival**: {female_survival:.1%}
- **Male Survival**: {male_survival:.1%}

## Age-Based Insights
- **Children (<18) Survival**: {child_survival:.1%}
- **Adults (18-59) Survival**: {adult_survival:.1%}
- **Seniors (60+) Survival**: {senior_survival_text}

## Family Impact
- **Small Families (2-4) Survival**: {small_family_survival:.1%}
- **Traveling Alone Survival**: {alone_survival:.1%}

## Key Findings

### 1. Social Class Was Decisive
First-class passengers had significantly higher survival rates, likely due to:
- Better cabin locations (closer to lifeboats)
- Priority in lifeboat access
- More assistance from crew

### 2. Gender Played Crucial Role
The "women and children first" protocol was strongly enforced, with female passengers having much higher survival rates.

### 3. Age Mattered
Children had better survival chances, though the pattern is complex and interacts with class and gender.

### 4. Family Size Had Mixed Impact
Small families had slightly better survival rates than those traveling alone or in very large families.

## Recommendations for Modeling
1. **Key Features**: Passenger class, gender, and fare are likely strong predictors
2. **Feature Engineering**: Create family size categories and age groups
3. **Missing Data**: Age has significant missing values that need careful handling
"""
        
        report_path = Path('docs/insights_report.md')
        report_path.write_text(insights)
        print(f"CREATED: {report_path}")
    
    def run_complete_analysis(self):
        """Run all analysis methods"""
        print("PHASE 3: Starting Comprehensive Data Exploration")
        print("=" * 50)
        
        self.create_demographic_analysis()
        self.create_fare_analysis()
        self.create_family_analysis()
        self.create_correlation_analysis()
        self.create_embarkation_analysis()
        self.generate_insights_report()
        
        print("\nPHASE 3 COMPLETE!")
        print("\nCREATED VISUALIZATIONS:")
        print("  - Survival by passenger class")
        print("  - Survival by gender") 
        print("  - Age distribution by survival")
        print("  - Fare analysis by class")
        print("  - Family size impact")
        print("  - Correlation matrix")
        print("  - Embarkation port analysis")
        print("\nNEXT STEPS:")
        print("  1. Review: docs/insights_report.md")
        print("  2. Check: presentation/assets/ for all plots")
        print("  3. Run: python preprocess_data.py")

def main():
    """Main exploration function"""
    explorer = DataExplorer()
    explorer.run_complete_analysis()

if __name__ == "__main__":
    main()