#!/usr/bin/env python3
"""
Test script for the Heart Disease Prediction System
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all API endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Heart Disease Prediction System API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test data initialization
    print("\n2. Testing data initialization...")
    try:
        response = requests.post(f"{base_url}/api/initialize-data")
        if response.status_code == 200:
            print("✅ Data initialization passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Data initialization failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data initialization error: {e}")
    
    # Test dataset info
    print("\n3. Testing dataset info...")
    try:
        response = requests.get(f"{base_url}/api/dataset-info")
        if response.status_code == 200:
            print("✅ Dataset info passed")
            data = response.json()
            if data['success']:
                print(f"   Dataset shape: {data['data']['shape']}")
                print(f"   Columns: {len(data['data']['columns'])}")
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Dataset info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dataset info error: {e}")
    
    # Test model training
    print("\n4. Testing model training...")
    try:
        training_data = {"model_type": "random_forest"}
        response = requests.post(
            f"{base_url}/api/train-model",
            json=training_data,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("✅ Model training passed")
            data = response.json()
            if data['success']:
                print(f"   Model: {data['model_type']}")
                print(f"   Accuracy: {data['accuracy']}%")
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Model training failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model training error: {e}")
    
    # Test prediction
    print("\n5. Testing prediction...")
    try:
        # Sample patient data
        patient_features = {
            "age": 55,
            "sex": 1,
            "cp": 1,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }
        
        prediction_data = {
            "model_type": "random_forest",
            "features": patient_features
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=prediction_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("✅ Prediction passed")
            data = response.json()
            if data['success']:
                print(f"   Prediction: {data['result_text']}")
                print(f"   Confidence: {data['probability']:.2%}")
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test visualizations
    print("\n6. Testing visualizations...")
    try:
        response = requests.get(f"{base_url}/api/visualizations")
        if response.status_code == 200:
            print("✅ Visualizations passed")
            data = response.json()
            if data['success']:
                print(f"   Available charts: {len(data['visualizations'])}")
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Visualizations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Visualizations error: {e}")
    
    # Test models list
    print("\n7. Testing models list...")
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            print("✅ Models list passed")
            data = response.json()
            if data['success']:
                print(f"   Available models: {len(data['models'])}")
                for model in data['models']:
                    print(f"     - {model['name']} ({model['type']})")
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Models list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models list error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

def test_frontend():
    """Test the frontend interface"""
    
    base_url = "http://localhost:5000"
    
    print("\n🌐 Testing Frontend Interface")
    print("=" * 50)
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Frontend accessible")
            if "Heart Disease Prediction System" in response.text:
                print("✅ Frontend content loaded correctly")
            else:
                print("⚠️  Frontend content may not be loading properly")
        else:
            print(f"❌ Frontend not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend test error: {e}")

if __name__ == "__main__":
    print("🚀 Starting system tests...")
    print("Make sure the Flask application is running on http://localhost:5000")
    print("You can start it with: python run.py")
    print()
    
    # Wait a moment for user to read
    time.sleep(2)
    
    try:
        test_api_endpoints()
        test_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
    
    print("\n💡 To run the application:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Start the server: python run.py")
    print("   3. Open browser: http://localhost:5000")
