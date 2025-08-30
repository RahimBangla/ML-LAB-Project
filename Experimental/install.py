#!/usr/bin/env python3
"""
Installation script for Heart Disease Prediction System
This script helps install dependencies step by step and checks for issues
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

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def install_basic_dependencies():
    """Install basic dependencies first"""
    print("\n📦 Installing basic dependencies...")
    
    # Install setuptools and wheel first
    if not run_command("pip install --upgrade setuptools wheel", "Upgrading setuptools and wheel"):
        return False
    
    # Install basic packages
    basic_packages = [
        "Flask",
        "Flask-CORS", 
        "pandas",
        "numpy",
        "requests"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    
    return True

def install_ml_dependencies():
    """Install machine learning dependencies"""
    print("\n🤖 Installing machine learning dependencies...")
    
    ml_packages = [
        "scikit-learn",
        "joblib",
        "plotly"
    ]
    
    for package in ml_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    
    return True

def install_optional_dependencies():
    """Install optional advanced dependencies"""
    print("\n🚀 Installing optional advanced dependencies...")
    
    optional_packages = [
        "xgboost",
        "tensorflow",
        "matplotlib",
        "seaborn"
    ]
    
    for package in optional_packages:
        print(f"\n📦 Trying to install {package}...")
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  {package} installation failed - this is optional, continuing...")
    
    return True

def test_imports():
    """Test if key packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("Flask", "flask"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("Scikit-learn", "sklearn"),
        ("Plotly", "plotly"),
        ("Joblib", "joblib")
    ]
    
    all_working = True
    
    for name, import_name in test_packages:
        try:
            __import__(import_name)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_working = False
    
    return all_working

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating necessary directories...")
    
    directories = ['data', 'models', 'uploads']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")
    
    return True

def main():
    """Main installation function"""
    print("🏥 Heart Disease Prediction System - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation cannot continue due to Python version incompatibility")
        return False
    
    # Install dependencies step by step
    if not install_basic_dependencies():
        print("\n⚠️  Basic dependencies installation had issues")
    
    if not install_ml_dependencies():
        print("\n⚠️  ML dependencies installation had issues")
    
    if not install_optional_dependencies():
        print("\n⚠️  Optional dependencies installation had issues")
    
    # Create directories
    create_directories()
    
    # Test imports
    print("\n" + "=" * 60)
    if test_imports():
        print("\n🎉 Installation completed successfully!")
        print("\n💡 You can now run the application with:")
        print("   python run.py")
        print("   or")
        print("   python app.py")
    else:
        print("\n⚠️  Installation completed with some issues")
        print("Some packages may not work properly")
    
    print("\n🌐 The application will be available at: http://localhost:5000")
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted by user")
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("\n💡 Try running: pip install --upgrade pip setuptools wheel")
