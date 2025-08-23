#!/usr/bin/env python3
"""
Demo script for the Heart Disease Prediction System
This script demonstrates the key features of the system
"""

import requests
import json
import time

def demo_heart_disease_prediction():
    """Demonstrate the complete heart disease prediction workflow"""
    
    base_url = "http://localhost:5000"
    
    print("🏥 Heart Disease Prediction System - Demo")
    print("=" * 60)
    print("This demo will show you how to:")
    print("1. Initialize the system with sample data")
    print("2. Train a machine learning model")
    print("3. Make predictions on patient data")
    print("4. View system analytics")
    print("=" * 60)
    print()
    
    # Step 1: Initialize sample data
    print("📊 Step 1: Initializing sample dataset...")
    try:
        response = requests.post(f"{base_url}/api/initialize-data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
        else:
            print("❌ Failed to initialize data")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    time.sleep(1)
    
    # Step 2: Check dataset info
    print("\n📈 Step 2: Checking dataset information...")
    try:
        response = requests.get(f"{base_url}/api/dataset-info")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                info = data['data']
                print(f"✅ Dataset loaded successfully!")
                print(f"   📁 Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
                print(f"   🎯 Target distribution: {info['target_distribution']}")
                print(f"   📋 Features: {', '.join(info['columns'][:-1])}")
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                return
        else:
            print("❌ Failed to get dataset info")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    time.sleep(1)
    
    # Step 3: Train a model
    print("\n🤖 Step 3: Training Random Forest model...")
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
                print(f"✅ Model trained successfully!")
                print(f"   🎯 Model Type: {data['model_type'].replace('_', ' ').title()}")
                print(f"   📊 Accuracy: {data['accuracy']}%")
                print(f"   📈 Confusion Matrix:")
                cm = data['confusion_matrix']
                print(f"      True Negatives: {cm[0][0]}")
                print(f"      False Positives: {cm[0][1]}")
                print(f"      False Negatives: {cm[1][0]}")
                print(f"      True Positives: {cm[1][1]}")
            else:
                print(f"❌ Training failed: {data.get('error', 'Unknown error')}")
                return
        else:
            print("❌ Failed to train model")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    time.sleep(1)
    
    # Step 4: Make predictions
    print("\n🔮 Step 4: Making predictions on sample patients...")
    
    # Sample patient cases
    patients = [
        {
            "name": "Patient A (High Risk)",
            "features": {
                "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 300,
                "fbs": 1, "restecg": 1, "thalach": 120, "exang": 1,
                "oldpeak": 3.0, "slope": 0, "ca": 2, "thal": 3
            }
        },
        {
            "name": "Patient B (Low Risk)",
            "features": {
                "age": 45, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
                "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
                "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 1
            }
        }
    ]
    
    for i, patient in enumerate(patients, 1):
        print(f"\n   👤 {patient['name']}:")
        try:
            prediction_data = {
                "model_type": "random_forest",
                "features": patient['features']
            }
            
            response = requests.post(
                f"{base_url}/api/predict",
                json=prediction_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    risk_level = "🔴 HIGH RISK" if data['prediction'] == 1 else "🟢 LOW RISK"
                    print(f"      {risk_level}: {data['result_text']}")
                    print(f"      📊 Confidence: {data['probability']:.1%}")
                    
                    # Show key risk factors
                    high_risk_factors = []
                    if patient['features']['age'] > 60:
                        high_risk_factors.append("Age > 60")
                    if patient['features']['trestbps'] > 140:
                        high_risk_factors.append("High blood pressure")
                    if patient['features']['chol'] > 240:
                        high_risk_factors.append("High cholesterol")
                    if patient['features']['cp'] == 3:
                        high_risk_factors.append("Asymptomatic chest pain")
                    
                    if high_risk_factors:
                        print(f"      ⚠️  Key risk factors: {', '.join(high_risk_factors)}")
                else:
                    print(f"      ❌ Prediction failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"      ❌ Prediction request failed")
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        time.sleep(0.5)
    
    # Step 5: Show system analytics
    print("\n📊 Step 5: Loading system analytics...")
    try:
        response = requests.get(f"{base_url}/api/visualizations")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Analytics loaded successfully!")
                print(f"   📊 Available visualizations: {len(data['visualizations'])}")
                print(f"   📈 Charts include: Target distribution, Age analysis, Correlation heatmap, Sex vs Target")
                print(f"   🌐 Open http://localhost:5000 in your browser to view interactive charts")
            else:
                print(f"❌ Analytics failed: {data.get('error', 'Unknown error')}")
        else:
            print("❌ Failed to load analytics")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 6: Show available models
    print("\n🤖 Step 6: Checking available models...")
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Models available: {len(data['models'])}")
                for model in data['models']:
                    print(f"   📁 {model['name']} ({model['type']})")
            else:
                print(f"❌ Failed to get models: {data.get('error', 'Unknown error')}")
        else:
            print("❌ Failed to get models list")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)
    print("💡 Next steps:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Explore the different tabs and features")
    print("   3. Try training different models")
    print("   4. Make predictions with your own data")
    print("   5. View the interactive visualizations")
    print("\n🚀 Happy predicting!")

if __name__ == "__main__":
    print("🚀 Starting Heart Disease Prediction System Demo...")
    print("Make sure the Flask application is running on http://localhost:5000")
    print("You can start it with: python run.py")
    print()
    
    try:
        demo_heart_disease_prediction()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n💡 Make sure the application is running:")
        print("   python run.py")
