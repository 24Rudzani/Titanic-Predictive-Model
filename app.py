"""
SUPER SIMPLE Titanic App - Direct data loading
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# MUST be first
st.set_page_config(
    page_title="Titanic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None

def load_data_directly():
    """Load data in the simplest way possible"""
    try:
        # Try absolute path
        st.session_state.df = pd.read_csv('data/raw/titanic.csv')
        st.session_state.data_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        # Create dummy data as fallback
        st.session_state.df = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 0],
            'Pclass': [1, 2, 3],
            'Sex': ['male', 'female', 'male'],
            'Age': [30, 25, 40],
            'Fare': [50, 30, 10]
        })
        st.session_state.data_loaded = False
        return False

def main():
    st.sidebar.title("Titanic Dashboard")
    
    # Load data button in sidebar
    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            if load_data_directly():
                st.sidebar.success("Data loaded!")
            else:
                st.sidebar.error("Using demo data")
    
    # Navigation
    page = st.sidebar.radio(
        "Go to:",
        ["Home", "View Data", "Analyze", "Predict"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data status: " + ("Loaded" if st.session_state.data_loaded else "Demo"))
    
    # Show selected page
    if page == "Home":
        show_home()
    elif page == "View Data":
        show_data()
    elif page == "Analyze":
        show_analysis()
    elif page == "Predict":
        show_prediction()

def show_home():
    # ========== IMAGE BEFORE TITLE ==========
    # Create columns for image + title
    col_img, col_title = st.columns([1, 3])
    
    with col_img:
        # Try to load your image from multiple possible paths
        image_loaded = False
        image_paths = [
            r"C:\Users\mothi\OneDrive\Desktop\Titanic Predictive Model\download.jpeg",
            "images/titanic.jpg",
            "titanic_image.jpg",
            "titanic.jpeg",
            "titanic.png",
            "download.jpeg"  # Try in current directory
        ]
        
        for img_path in image_paths:
            try:
                if Path(img_path).exists():
                    st.image(img_path, width=250)
                    image_loaded = True
                    break
            except Exception as e:
                continue  # Try next path
        
        # If no image found, show fallback
        if not image_loaded:
            st.write("🚢")  # Fallback emoji
    
    with col_title:
        st.title("Titanic Survival Analysis")
    
    st.markdown("---")
    
    # Rest of your home page content...
    if not st.session_state.data_loaded:
        st.warning("⚠️ Using demo data. Click 'Load Data' in sidebar to load real Titanic data.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Project", "Titanic ML")
    
    with col2:
        st.metric("Status", "Ready")
    
    with col3:
        if st.session_state.data_loaded:
            st.metric("Data", "Real")
        else:
            st.metric("Data", "Demo")
    
    st.markdown("---")
    
    st.subheader("About This Project")
    st.markdown("""
    This dashboard analyzes passenger data from the RMS Titanic to:
    
    - 📊 Explore passenger demographics and survival patterns
    - 🤖 Predict survival probabilities using machine learning
    - 📈 Understand factors that influenced survival rates
    - 🎯 Provide insights into historical events
    
    **Key Findings:**
    - Passenger class was a major survival factor
    - "Women and children first" protocol was strongly followed
    - Age and family size also played significant roles
    """)
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.success(f"✅ Real data loaded: {len(st.session_state.df)} passengers")

def show_data():
    st.title("📋 Titanic Passenger Data")
    
    if st.session_state.df is None:
        st.error("No data available. Click 'Load Data' in sidebar.")
        return
    
    st.write(f"**Total passengers:** {len(st.session_state.df)}")
    
    # Show data
    st.dataframe(st.session_state.df, use_container_width=True)
    
    # Basic stats
    st.subheader("Basic Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Survived' in st.session_state.df.columns:
            survived = st.session_state.df['Survived'].sum()
            total = len(st.session_state.df)
            st.metric("Survival Rate", f"{survived/total*100:.1f}%")
    
    with col2:
        if 'Pclass' in st.session_state.df.columns:
            classes = st.session_state.df['Pclass'].value_counts()
            st.write("**Passenger Classes:**")
            for cls, count in classes.items():
                st.write(f"  Class {cls}: {count}")
    
    with col3:
        if 'Sex' in st.session_state.df.columns:
            genders = st.session_state.df['Sex'].value_counts()
            st.write("**Gender Distribution:**")
            for gender, count in genders.items():
                st.write(f"  {gender}: {count}")

def show_analysis():
    st.title("📈 Data Analysis")
    
    if st.session_state.df is None:
        st.error("No data available. Click 'Load Data' in sidebar.")
        return
    
    df = st.session_state.df
    
    st.subheader("Survival Analysis")
    
    # Survival by class
    if 'Pclass' in df.columns and 'Survived' in df.columns:
        st.write("**Survival by Passenger Class:**")
        class_survival = df.groupby('Pclass')['Survived'].mean()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for pclass, rate in class_survival.items():
                st.write(f"Class {pclass}: {rate:.1%}")
        
        with col2:
            st.bar_chart(class_survival)
    
    # Survival by gender
    if 'Sex' in df.columns and 'Survived' in df.columns:
        st.write("**Survival by Gender:**")
        gender_survival = df.groupby('Sex')['Survived'].mean()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for gender, rate in gender_survival.items():
                st.write(f"{gender}: {rate:.1%}")
        
        with col2:
            st.bar_chart(gender_survival)
    
    # Age analysis
    if 'Age' in df.columns:
        st.write("**Age Distribution:**")
        age_data = df['Age'].dropna()
        if len(age_data) > 0:
            st.write(f"Average age: {age_data.mean():.1f} years")
            st.write(f"Youngest: {age_data.min():.1f} years")
            st.write(f"Oldest: {age_data.max():.1f} years")

def show_prediction():
    st.title("🤖 Survival Prediction")
    
    st.markdown("Enter passenger details to estimate survival probability:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3])
            age = st.slider("Age", 0, 80, 30)
        
        with col2:
            sex = st.selectbox("Gender", ["male", "female"])
            fare = st.slider("Fare ($)", 0, 200, 50)
        
        submitted = st.form_submit_button("Estimate Survival")
        
        if submitted:
            # Simple prediction logic
            survival_chance = 0.5  # Base chance
            
            # Adjust based on class
            if pclass == 1:
                survival_chance += 0.3
            elif pclass == 3:
                survival_chance -= 0.2
            
            # Adjust based on gender
            if sex == "female":
                survival_chance += 0.3
            else:
                survival_chance -= 0.2
            
            # Adjust based on age
            if age < 18:
                survival_chance += 0.1
            
            # Adjust based on fare
            if fare > 100:
                survival_chance += 0.1
            
            # Ensure bounds
            survival_chance = max(0.05, min(0.95, survival_chance))
            
            # Display results
            st.markdown("---")
            st.subheader("Estimation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Survival Chance", f"{survival_chance:.1%}")
            
            with col2:
                status = "Likely Survived" if survival_chance > 0.5 else "Likely Did Not Survive"
                st.metric("Prediction", status)
            
            # Explanation
            st.subheader("Factors Considered")
            factors = []
            
            if pclass == 1:
                factors.append("✅ First class: Better access to lifeboats")
            elif pclass == 3:
                factors.append("❌ Third class: Limited access to lifeboats")
            
            if sex == "female":
                factors.append("✅ Female: 'Women and children first' protocol")
            else:
                factors.append("❌ Male: Lower priority in rescue")
            
            if age < 18:
                factors.append("✅ Child: Higher rescue priority")
            
            if fare > 100:
                factors.append("✅ High fare: Indicates wealthier passenger")
            
            for factor in factors:
                st.write(factor)

# Run the app
if __name__ == "__main__":
    # Try to load data automatically
    if not st.session_state.data_loaded:
        load_data_directly()
    
    main()