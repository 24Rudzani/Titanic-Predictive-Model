"""
STEP 5: Model Training and Evaluation
Train multiple machine learning models and evaluate their performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Train and evaluate multiple machine learning models for Titanic survival prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load processed training and testing data"""
        print("DATA: Loading processed datasets...")
        
        processed_dir = Path('data/processed')
        
        # Load scaled features (better for most algorithms)
        self.X_train = pd.read_csv(processed_dir / 'X_train_scaled.csv')
        self.X_test = pd.read_csv(processed_dir / 'X_test_scaled.csv')
        self.y_train = pd.read_csv(processed_dir / 'y_train.csv').squeeze()
        self.y_test = pd.read_csv(processed_dir / 'y_test.csv').squeeze()
        
        # Load feature names
        self.feature_names = self.X_train.columns.tolist()
        
        print(f"TRAINING DATA: {self.X_train.shape}")
        print(f"TESTING DATA: {self.X_test.shape}")
        print(f"FEATURES: {len(self.feature_names)}")
        
    def initialize_models(self):
        """Initialize multiple machine learning models"""
        print("MODELS: Initializing machine learning models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'Support Vector Machine': SVC(random_state=self.random_state, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state)
        }
        
        print(f"INITIALIZED: {len(self.models)} models for comparison")
    
    def train_and_evaluate_models(self):
        """Train all models and evaluate their performance"""
        print("\nTRAINING: Training and evaluating all models...")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Calculate AUC-ROC if probability predictions are available
            auc_roc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  {name}:")
            print(f"    Accuracy: {accuracy:.3f}")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            print(f"    CV Score: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
            if auc_roc:
                print(f"    AUC-ROC: {auc_roc:.3f}")
            print()
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best performing model"""
        print("HYPERPARAMETER TUNING: Optimizing best model...")
        
        if self.best_model == 'Random Forest':
            print("Tuning Random Forest hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            # Update with best parameters
            self.models['Random Forest (Tuned)'] = grid_search.best_estimator_
            self.results['Random Forest (Tuned)'] = self.evaluate_single_model(
                grid_search.best_estimator_, 'Random Forest (Tuned)'
            )
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        elif self.best_model == 'Gradient Boosting':
            print("Tuning Gradient Boosting hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5]
            }
            
            gb = GradientBoostingClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            self.models['Gradient Boosting (Tuned)'] = grid_search.best_estimator_
            self.results['Gradient Boosting (Tuned)'] = self.evaluate_single_model(
                grid_search.best_estimator_, 'Gradient Boosting (Tuned)'
            )
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    def evaluate_single_model(self, model, name):
        """Evaluate a single model and return metrics"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        return {
            'model': model,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def create_model_comparison_plot(self):
        """Create visualization comparing model performance"""
        print("VISUALIZATION: Creating model comparison plot...")
        
        # Prepare data for plotting
        models_list = list(self.results.keys())
        accuracy_scores = [self.results[model]['accuracy'] for model in models_list]
        f1_scores = [self.results[model]['f1_score'] for model in models_list]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.barh(models_list, accuracy_scores, color='skyblue')
        ax1.set_xlabel('Accuracy Score')
        ax1.set_title('Model Comparison - Accuracy')
        ax1.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # F1-Score comparison
        bars2 = ax2.barh(models_list, f1_scores, color='lightcoral')
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Model Comparison - F1-Score')
        ax2.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('presentation/assets/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Model comparison visualization")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("FEATURE IMPORTANCE: Analyzing important features...")
        
        # Get the best tree-based model
        best_model_name = self.best_model
        if '(Tuned)' in self.best_model:
            best_model_name = self.best_model.replace(' (Tuned)', '')
        
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            model = self.results[self.best_model]['model']
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                # Plot top 15 features
                plt.figure(figsize=(10, 8))
                plt.barh(feature_importance.tail(15)['feature'], 
                        feature_importance.tail(15)['importance'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importance - {self.best_model}')
                plt.tight_layout()
                plt.savefig('presentation/assets/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("CREATED: Feature importance plot")
                
                # Print top 10 features
                print("\nTOP 10 MOST IMPORTANT FEATURES:")
                top_features = feature_importance.tail(10)
                for _, row in top_features.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the best model"""
        print("CONFUSION MATRIX: Creating for best model...")
        
        best_result = self.results[self.best_model]
        cm = confusion_matrix(self.y_test, best_result['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Did Not Survive', 'Survived'],
                   yticklabels=['Did Not Survive', 'Survived'])
        plt.title(f'Confusion Matrix - {self.best_model}\nAccuracy: {best_result["accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('presentation/assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("CREATED: Confusion matrix")
    
    def save_models(self):
        """Save trained models to disk"""
        print("SAVING: Saving trained models...")
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save all models
        for name, result in self.results.items():
            model = result['model']
            filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
            joblib.dump(model, models_dir / filename)
        
        # Save the best model separately for easy access
        best_model = self.results[self.best_model]['model']
        joblib.dump(best_model, models_dir / 'best_model.pkl')
        
        # Save results summary
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1-Score': [self.results[model]['f1_score'] for model in self.results.keys()],
            'AUC-ROC': [self.results[model]['auc_roc'] or 0 for model in self.results.keys()]
        })
        results_df.to_csv(models_dir / 'model_results.csv', index=False)
        
        print(f"SAVED: {len(self.results)} models to models/ directory")
        print(f"BEST MODEL: {self.best_model} (Accuracy: {self.best_score:.3f})")
    
    def generate_modeling_report(self):
        """Generate comprehensive modeling report"""
        print("REPORT: Generating modeling report...")
        
        # Create results table
        results_table = "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        results_table += "|-------|----------|-----------|--------|----------|---------|\n"
        
        for model_name in sorted(self.results.keys(), 
                               key=lambda x: self.results[x]['accuracy'], reverse=True):
            result = self.results[model_name]
            results_table += f"| {model_name} | {result['accuracy']:.3f} | {result['precision']:.3f} | {result['recall']:.3f} | {result['f1_score']:.3f} | {result['auc_roc'] or 0:.3f} |\n"
        
        report = f"""# Model Training Report

## Best Performing Model
- **Model**: {self.best_model}
- **Accuracy**: {self.results[self.best_model]['accuracy']:.3f}
- **Precision**: {self.results[self.best_model]['precision']:.3f}
- **Recall**: {self.results[self.best_model]['recall']:.3f}
- **F1-Score**: {self.results[self.best_model]['f1_score']:.3f}

## Model Performance Comparison

{results_table}

## Key Findings

### 1. Best Performing Algorithm
The {self.best_model} achieved the highest accuracy of {self.best_score:.3f} on the test set.

### 2. Model Strengths
- **Tree-based models** (Random Forest, Gradient Boosting) typically perform well on tabular data
- **Logistic Regression** provides good interpretability with reasonable performance
- **Ensemble methods** often outperform single models

### 3. Important Features
Based on feature importance analysis, the most predictive features align with historical knowledge:
- Passenger class and fare (proxy for socioeconomic status)
- Gender (reflecting "women and children first" protocol)
- Age and family characteristics

## Recommendations

1. **Production Model**: Use {self.best_model} for predictions
2. **Interpretability**: Consider Logistic Regression if model explainability is important
3. **Monitoring**: Track model performance over time for potential retraining

## Next Steps
1. Integrate the best model into the Streamlit application
2. Create prediction interface for new passenger data
3. Develop model monitoring system
"""
        
        report_path = Path('docs/modeling_report.md')
        report_path.write_text(report)
        print(f"CREATED: {report_path}")
    
    def run_complete_training(self):
        """Run the complete model training pipeline"""
        print("PHASE 5: Starting Model Training Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Initialize and train models
        self.initialize_models()
        self.train_and_evaluate_models()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Create visualizations
        self.create_model_comparison_plot()
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        
        # Save results
        self.save_models()
        self.generate_modeling_report()
        
        print("\nPHASE 5 COMPLETE!")
        print(f"\nBEST MODEL: {self.best_model} (Accuracy: {self.best_score:.3f})")
        print("\nCREATED OUTPUTS:")
        print("  - Trained models: models/ directory")
        print("  - Model comparison: presentation/assets/model_comparison.png")
        print("  - Feature importance: presentation/assets/feature_importance.png")
        print("  - Confusion matrix: presentation/assets/confusion_matrix.png")
        print("  - Modeling report: docs/modeling_report.md")
        print("\nNEXT STEPS:")
        print("  1. Review: docs/modeling_report.md")
        print("  2. Run: python app.py to start the Streamlit app")

def main():
    """Main training function"""
    trainer = ModelTrainer()
    trainer.run_complete_training()

if __name__ == "__main__":
    main()