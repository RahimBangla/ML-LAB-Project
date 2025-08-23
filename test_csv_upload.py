#!/usr/bin/env python3
"""
Test script for CSV upload functionality
This script tests the new CSV upload and data management features
"""

import requests
import json
import pandas as pd
import os

def test_csv_upload():
    """Test the CSV upload functionality"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing CSV Upload Functionality")
    print("=" * 50)
    
    # Test 1: Download template
    print("\n1. Testing template download...")
    try:
        response = requests.get(f"{base_url}/api/download-template")
        if response.status_code == 200:
            print("✅ Template download successful")
            
            # Save template locally for testing
            with open('test_template.csv', 'wb') as f:
                f.write(response.content)
            print("   📁 Template saved as 'test_template.csv'")
        else:
            print(f"❌ Template download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template download error: {e}")
        return False
    
    # Test 2: Create a test dataset
    print("\n2. Creating test dataset...")
    test_data = {
        'age': [45, 52, 58, 63, 70],
        'sex': [0, 1, 0, 1, 0],
        'cp': [0, 1, 2, 3, 0],
        'trestbps': [110, 125, 135, 150, 160],
        'chol': [180, 200, 220, 250, 300],
        'fbs': [0, 0, 1, 1, 0],
        'restecg': [0, 1, 0, 1, 2],
        'thalach': [170, 160, 150, 140, 130],
        'exang': [0, 1, 1, 1, 1],
        'oldpeak': [0.5, 1.0, 1.5, 2.0, 3.0],
        'slope': [2, 1, 0, 0, 0],
        'ca': [0, 0, 1, 2, 3],
        'thal': [1, 2, 2, 3, 3],
        'target': [0, 0, 1, 1, 1]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_dataset.csv', index=False)
    print("✅ Test dataset created: 'test_dataset.csv'")
    
    # Test 3: Upload test dataset
    print("\n3. Testing dataset upload...")
    try:
        with open('test_dataset.csv', 'rb') as f:
            files = {'file': ('test_dataset.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/api/upload-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Dataset upload successful")
                print(f"   📊 Message: {data['message']}")
                if 'dataset_info' in data:
                    info = data['dataset_info']
                    print(f"   📈 Shape: {info['shape']}")
                    print(f"   🎯 Target distribution: {info['target_distribution']}")
            else:
                print(f"❌ Upload failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Upload request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    # Test 4: Verify dataset info
    print("\n4. Verifying dataset info...")
    try:
        response = requests.get(f"{base_url}/api/dataset-info")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                info = data['data']
                print("✅ Dataset info retrieved successfully")
                print(f"   📊 Shape: {info['shape']}")
                print(f"   🎯 Target distribution: {info['target_distribution']}")
                print(f"   📋 Columns: {len(info['columns'])}")
                
                if 'feature_stats' in info:
                    stats = info['feature_stats']
                    print(f"   📈 Age range: {stats['age']['min']}-{stats['age']['max']}")
                    print(f"   💓 Blood pressure range: {stats['trestbps']['min']}-{stats['trestbps']['max']}")
            else:
                print(f"❌ Dataset info failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Dataset info request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dataset info error: {e}")
        return False
    
    # Test 5: Train a model on uploaded data
    print("\n5. Testing model training on uploaded data...")
    try:
        training_data = {"model_type": "random_forest"}
        response = requests.post(
            f"{base_url}/api/train-model",
            json=training_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Model training successful on uploaded data")
                print(f"   🎯 Model: {data['model_type']}")
                print(f"   📊 Accuracy: {data['accuracy']}%")
                print(f"   📈 Training samples: {data['training_samples']}")
                print(f"   🧪 Testing samples: {data['testing_samples']}")
            else:
                print(f"❌ Training failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Training request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    # Test 6: Make prediction with uploaded data
    print("\n6. Testing prediction with uploaded data...")
    try:
        prediction_data = {
            "model_type": "random_forest",
            "features": {
                "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 220,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
            }
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=prediction_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Prediction successful with uploaded data")
                print(f"   🔮 Result: {data['result_text']}")
                print(f"   📊 Confidence: {data['probability']:.1%}")
                print(f"   ⚠️  Risk Level: {data['risk_level']}")
                
                if data['recommendations']:
                    print(f"   💡 Recommendations: {len(data['recommendations'])} provided")
            else:
                print(f"❌ Prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    for file in ['test_template.csv', 'test_dataset.csv']:
        if os.path.exists(file):
            os.remove(file)
            print(f"   ✅ Removed {file}")
    
    print("\n" + "=" * 50)
    print("🎉 CSV Upload Testing Completed Successfully!")
    print("=" * 50)
    print("✅ All functionality working correctly:")
    print("   📥 Template download")
    print("   📤 Dataset upload")
    print("   🔍 Data validation")
    print("   🤖 Model training")
    print("   🔮 Predictions")
    print("   📊 Data analytics")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting CSV Upload Functionality Test...")
    print("Make sure the Flask application is running on http://localhost:5000")
    print("You can start it with: python run.py")
    print()
    
    try:
        success = test_csv_upload()
        if success:
            print("\n💡 The CSV upload functionality is working perfectly!")
            print("You can now upload your own heart disease datasets.")
        else:
            print("\n❌ Some tests failed. Check the error messages above.")
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("\n💡 Make sure the application is running:")
        print("   python run.py")
