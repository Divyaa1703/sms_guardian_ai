#!/usr/bin/env python3
"""
Setup script for SMS Spam Classifier
This script helps set up the environment and download required NLTK data
"""

import subprocess
import sys
import nltk
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False
    return True

def check_model_files():
    """Check if required model files exist"""
    required_files = ['model.pkl', 'vectorizer.pkl', 'spam.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing required files: {', '.join(missing_files)}")
        print("Please ensure you have run the Jupyter notebook to generate the model files.")
        return False
    else:
        print("✓ All required model files found!")
        return True

def main():
    """Main setup function"""
    print("Setting up SMS Spam Classifier...")
    print("=" * 50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    # Check model files
    if not check_model_files():
        success = False
    
    print("=" * 50)
    if success:
        print("✓ Setup completed successfully!")
        print("You can now run the app with: streamlit run app.py")
    else:
        print("✗ Setup completed with errors. Please check the messages above.")

if __name__ == "__main__":
    main()
