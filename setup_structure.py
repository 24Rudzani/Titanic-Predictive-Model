"""
STEP 1: Create the project structure inside 'Titanic Predictive Model'
Without emojis to avoid encoding issues
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete folder structure"""
    
    print("SETUP: Setting up Titanic Predictive Model Project...")
    print("=" * 50)
    
    # Define all the folders we need
    folders = [
        # Data directories
        'data/raw',
        'data/processed',
        
        # Source code packages
        'src/data',
        'src/features', 
        'src/models',
        'src/visualization',
        
        # Streamlit app
        'app/pages',
        'app/assets',
        
        # Presentation
        'presentation/slides',
        'presentation/assets',
        
        # Additional directories
        'notebooks',
        'models',
        'tests',
        'docs',
        'config'
    ]
    
    # Create all folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"CREATED: {folder}/")
    
    # Create __init__.py files for Python packages
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/features/__init__.py',
        'src/models/__init__.py', 
        'src/visualization/__init__.py',
        'app/__init__.py',
        'app/pages/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).write_text('')
        print(f"FILE CREATED: {init_file}")
    
    print("\nSUCCESS: Project structure created successfully!")
    return True

def create_requirements_file():
    """Create requirements.txt with necessary packages"""
    
    requirements = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
python-pptx>=0.6.21
jupyter>=1.0.0
joblib>=1.3.0
"""
    
    Path('requirements.txt').write_text(requirements)
    print("CREATED: requirements.txt")
    
    return True

def main():
    """Main setup function"""
    create_project_structure()
    create_requirements_file()
    
    print("\nPHASE 1 COMPLETE!")
    print("\nNEXT STEPS:")
    print("   1. Run: pip install -r requirements.txt")
    print("   2. Run: python download_data.py")
    print("   3. Start exploring the data!")

if __name__ == "__main__":
    main()