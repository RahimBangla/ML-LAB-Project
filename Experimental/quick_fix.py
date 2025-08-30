#!/usr/bin/env python3
"""
Quick fix script for the Heart Disease Prediction System
This script resolves common installation issues
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def quick_fix():
    """Run quick fixes for common issues"""
    print("🚀 Quick Fix for Heart Disease Prediction System")
    print("=" * 60)
    
    # Fix 1: Upgrade pip, setuptools, and wheel
    print("\n📦 Fix 1: Upgrading core build tools...")
    if run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        print("✅ pip upgraded successfully")
    
    if run_command("pip install --upgrade setuptools wheel", "Upgrading setuptools and wheel"):
        print("✅ setuptools and wheel upgraded successfully")
    
    # Fix 2: Install basic dependencies
    print("\n📦 Fix 2: Installing basic dependencies...")
    basic_packages = [
        "Flask",
        "Flask-CORS",
        "pandas",
        "numpy",
        "requests"
    ]
    
    for package in basic_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️  {package} installation failed, but continuing...")
    
    # Fix 3: Try to install ML packages
    print("\n🤖 Fix 3: Installing machine learning packages...")
    ml_packages = [
        "scikit-learn",
        "joblib",
        "plotly"
    ]
    
    for package in ml_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️  {package} installation failed, but continuing...")
    
    # Fix 4: Create directories
    print("\n📁 Fix 4: Creating necessary directories...")
    directories = ['data', 'models', 'uploads']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")
    
    print("\n" + "=" * 60)
    print("🎉 Quick fix completed!")
    print("=" * 60)
    
    # Test if we can run the app
    print("\n🧪 Testing if the application can start...")
    try:
        # Try to import key packages
        import flask
        import pandas
        import numpy
        print("✅ All basic packages imported successfully!")
        
        print("\n💡 You can now try running the application:")
        print("   python app.py")
        print("   or")
        print("   python run.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("\n💡 Try running the full installation script:")
        print("   python install.py")
        return False

if __name__ == "__main__":
    try:
        quick_fix()
    except KeyboardInterrupt:
        print("\n🛑 Quick fix interrupted by user")
    except Exception as e:
        print(f"\n❌ Quick fix failed: {e}")
        print("\n💡 Try running: pip install --upgrade pip setuptools wheel")
