🚢 Titanic Survival Predictive Model

A comprehensive machine learning project that analyzes and predicts survival outcomes of Titanic passengers. This end-to-end data science solution includes data exploration, preprocessing, model training, and an interactive dashboard for visualization and prediction.

[![Dashboard Live](https://img.shields.io/badge/Dashboard-Live-brightgreen)]
[![Accuracy](https://img.shields.io/badge/Accuracy-83.2%25-blue)]
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)]
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)]

🌟 Live Demo

- Interactive Dashboard: https://titanic-predictive-model-eshtzwblq5gjxv8duz8qff.streamlit.app/
- GitHub Repository: https://github.com/24Rudzani/Titanic-Predictive-Model

📊 Project Overview

This project analyzes the famous Titanic dataset to understand factors that influenced passenger survival during the 1912 tragedy. Using machine learning techniques, we build predictive models and create an interactive dashboard for exploring the data and making predictions.

Key Features:
- Data Analysis: Comprehensive exploration of passenger demographics
- Machine Learning: Six ML models compared with 83.2% top accuracy
- Interactive Dashboard: Real-time predictions and visualizations
- Feature Engineering: Advanced preprocessing techniques
- Historical Insights: Data-driven validation of historical patterns

🏗️ Project Architecture

Titanic-Predictive-Model/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── raw/                 # Raw Titanic dataset
│   └── processed/           # Preprocessed datasets
│
├── models/                  # Trained ML models
├── notebooks/               # Jupyter notebooks for exploration
├── presentation/            # Visualization assets
├── docs/                    # Documentation and reports
│
├── download_data.py         # Data download script
├── explore_data.py          # Data exploration and visualization
├── preprocess_data.py       # Data preprocessing pipeline
├── train_model.py           # Model training and evaluation
├── check_data.py           # File validation
└── debug_data.py           # Debugging utilities

🚀 Quick Start

Prerequisites
- Python 3.8+
- pip package manager

Installation

1. Clone the repository:

git clone https://github.com/24Rudzani/Titanic-Predictive-Model.git
cd Titanic-Predictive-Model


2. Install dependencies:

pip install -r requirements.txt


3. Set up the project structure:

python setup_structure.py


4. Download and prepare the data:

python download_data.py
python explore_data.py
python preprocess_data.py
python train_model.py


5. Run the dashboard:

streamlit run app.py



📈 Machine Learning Pipeline

1. Data Collection
- Downloads Titanic dataset from reliable sources
- 891 passenger records with 12 features
- Handles download failures with multiple source fallbacks

2. Data Preprocessing
- Missing Value Imputation: Age, Fare, Embarked
- Feature Engineering:
  - Title extraction from names
  - Family size categories
  - Age and fare grouping
  - Deck extraction from cabin numbers
- Encoding: One-hot encoding for categorical variables
- Scaling: Standardization of numerical features

3. Model Training
Six machine learning models trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 83.2% | 80.0% | 75.4% | 77.6% |
| Support Vector Machine | 83.2% | 80.0% | 75.4% | 77.6% |
| Gradient Boosting | 81.6% | 78.9% | 73.2% | 75.9% |
| Random Forest | 79.3% | 76.5% | 70.7% | 73.5% |
| K-Nearest Neighbors | 78.7% | 75.8% | 70.1% | 72.9% |
| Decision Tree | 77.6% | 74.5% | 68.9% | 71.6% |

4. Key Findings
- Gender: 74% female vs. 19% male survival rate
- Passenger Class: First class had 62% survival vs. third class 24%
- Age: Children (<12) had 59% survival rate
- Family Size: Small families (2-4) had optimal survival rates

🎮 Dashboard Features

📋 Home Page
- Project overview and statistics
- Image display with fallback options
- Data loading status indicators

📊 View Data
- Interactive dataframe display
- Basic statistics and metrics
- Passenger class and gender distributions

📈 Analyze
- Survival analysis by class and gender
- Age distribution visualizations
- Interactive bar charts and statistics

🤖 Predict
- Rule-based survival probability calculator
- Real-time prediction based on passenger features
- Factor analysis and explanation of predictions

🔧 Technical Implementation

Tech Stack
- Frontend: Streamlit for interactive dashboard
- Backend: Scikit-learn for machine learning models
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Plotly
- Deployment: Streamlit Cloud

Key Scripts
- app.py: Main Streamlit application with interactive features
- download_data.py: Robust data download with multiple fallback sources
- explore_data.py: Comprehensive data analysis and visualization
- preprocess_data.py: Complete data preprocessing pipeline
- train_model.py: Multi-model training and evaluation system

📊 Results & Insights

Historical Validation
✅ "Women and children first" protocol confirmed by data (74% female survival vs. 19% male)  
✅ Class privilege significantly affected survival chances (62% 1st class vs. 24% 3rd class)  
✅ Wealth (via fare) correlated with better outcomes  

Unexpected Discoveries
• Family size sweet spot of 2-4 members had optimal survival  
• Passenger titles were stronger predictors than raw age  
• Complex interactions between features revealed through feature engineering  

📈 Performance Metrics

- Best Model: Logistic Regression / Support Vector Machine
- Accuracy: 83.2%
- Precision: 80.0%
- Recall: 75.4%
- F1-Score: 77.6%
- Cross-Validation: 5-fold validation with consistent performance

🚢 Usage Examples

Making Predictions


📈 Machine Learning Pipeline

1. Data Collection
- Downloads Titanic dataset from reliable sources
- 891 passenger records with 12 features
- Handles download failures with multiple source fallbacks

2. Data Preprocessing
- Missing Value Imputation: Age, Fare, Embarked
- Feature Engineering:
  - Title extraction from names
  - Family size categories
  - Age and fare grouping
  - Deck extraction from cabin numbers
- Encoding: One-hot encoding for categorical variables
- Scaling: Standardization of numerical features

3. Model Training
Six machine learning models trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 83.2% | 80.0% | 75.4% | 77.6% |
| Support Vector Machine | 83.2% | 80.0% | 75.4% | 77.6% |
| Gradient Boosting | 81.6% | 78.9% | 73.2% | 75.9% |
| Random Forest | 79.3% | 76.5% | 70.7% | 73.5% |
| K-Nearest Neighbors | 78.7% | 75.8% | 70.1% | 72.9% |
| Decision Tree | 77.6% | 74.5% | 68.9% | 71.6% |

4. Key Findings
- Gender: 74% female vs. 19% male survival rate
- Passenger Class: First class had 62% survival vs. third class 24%
- Age: Children (<12) had 59% survival rate
- Family Size: Small families (2-4) had optimal survival rates

🎮 Dashboard Features

📋 Home Page
- Project overview and statistics
- Image display with fallback options
- Data loading status indicators

📊 View Data
- Interactive dataframe display
- Basic statistics and metrics
- Passenger class and gender distributions

📈 Analyze
- Survival analysis by class and gender
- Age distribution visualizations
- Interactive bar charts and statistics

🤖 Predict
- Rule-based survival probability calculator
- Real-time prediction based on passenger features
- Factor analysis and explanation of predictions

🔧 Technical Implementation

Tech Stack
- Frontend: Streamlit for interactive dashboard
- Backend: Scikit-learn for machine learning models
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Plotly
- Deployment: Streamlit Cloud

Key Scripts
- app.py: Main Streamlit application with interactive features
- download_data.py: Robust data download with multiple fallback sources
- explore_data.py: Comprehensive data analysis and visualization
- preprocess_data.py: Complete data preprocessing pipeline
- train_model.py: Multi-model training and evaluation system

📊 Results & Insights

Historical Validation
✅ "Women and children first" protocol confirmed by data (74% female survival vs. 19% male)  
✅ Class privilege significantly affected survival chances (62% 1st class vs. 24% 3rd class)  
✅ Wealth (via fare) correlated with better outcomes  

Unexpected Discoveries
• Family size sweet spot of 2-4 members had optimal survival  
• Passenger titles were stronger predictors than raw age  
• Complex interactions between features revealed through feature engineering  

📈 Performance Metrics

- Best Model: Logistic Regression / Support Vector Machine
- Accuracy: 83.2%
- Precision: 80.0%
- Recall: 75.4%
- F1-Score: 77.6%
- Cross-Validation: 5-fold validation with consistent performance

🚢 Usage Examples

Making Predictions

Example passenger data
passenger_data = {
'Pclass': 1, # First class
'Sex': 'female', # Female passenger
'Age': 28, # 28 years old
'Fare': 150 # $150 fare
}

Model predicts ~85% survival probability


Data Exploration

Load and explore data
import pandas as pd
df = pd.read_csv('data/processed/titanic_processed.csv')
print(f"Survival rate: {df['Survived'].mean():.1%}")
print(f"Average age: {df['Age'].mean():.1f} years")


📚 Documentation

- docs/data_summary.md: Complete dataset overview
- docs/insights_report.md: Key findings and historical insights
- docs/preprocessing_report.md: Data preprocessing steps and decisions
- docs/modeling_report.md: Model training results and comparisons

🛠️ Troubleshooting

Common issues and solutions:

1. Data loading failed:

python debug_data.py
python download_data.py # Re-download data


2. Missing dependencies:

pip install -r requirements.txt --upgrade


3. Streamlit app not loading:

streamlit cache clear
streamlit run app.py


🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

- Titanic dataset providers and maintainers
- Streamlit team for the amazing framework
- Scikit-learn developers for the robust ML library
- Open source community for invaluable tools and resources

👨‍💻 Author

Rudzani Junior Munyai
- GitHub: https://github.com/24Rudzani/Titanic-Predictive-Model
- Project: Titanic Predictive Model
- Live Demo: https://titanic-predictive-model-eshtzwblq5gjxv8duz8qff.streamlit.app/

---

⭐ If you find this project useful, please give it a star! ⭐

"The story of the Titanic continues to teach us about survival, human decision-making, and the importance of data in understanding historical events."
