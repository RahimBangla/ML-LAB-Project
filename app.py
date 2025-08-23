from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Model saving will be disabled.")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML models will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost model will be disabled.")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural network model will be disabled.")

try:
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Visualizations will be disabled.")

app = Flask(__name__)
CORS(app)

# Global variables
MODELS_DIR = 'models'
DATA_DIR = 'data'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create directories if they don't exist
for directory in [MODELS_DIR, DATA_DIR, UPLOAD_FOLDER]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Sample heart disease dataset (you can replace this with your own data)
SAMPLE_DATA = {
    'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
    'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    'cp': [3, 2, 1, 1, 0, 0, 1, 1, 2, 2],
    'trestbps': [145, 130, 130, 120, 120, 110, 140, 120, 172, 150],
    'chol': [233, 250, 204, 236, 354, 177, 288, 263, 199, 126],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    'restecg': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    'thalach': [150, 187, 172, 178, 163, 165, 133, 173, 162, 173],
    'exang': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 2.6, 4.0, 0.0, 0.5, 1.6],
    'slope': [0, 0, 2, 2, 2, 2, 3, 1, 1, 2],
    'ca': [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    'thal': [1, 2, 2, 2, 2, 2, 3, 2, 2, 2],
    'target': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataset(df):
    """Validate uploaded dataset structure"""
    required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if target column has binary values
    if not set(df['target'].unique()).issubset({0, 1}):
        return False, "Target column must contain only 0 and 1 values"
    
    # Check for missing values
    if df.isnull().any().any():
        return False, "Dataset contains missing values"
    
    # Check data types
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column {col} must be numeric"
    
    return True, "Dataset is valid"

def preprocess_dataset(df):
    """Preprocess the dataset for training"""
    # Convert target to int if it's not already
    df['target'] = df['target'].astype(int)
    
    # Ensure all feature columns are numeric
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with any NaN values
    df = df.dropna()
    
    return df

# Initialize sample dataset
def initialize_sample_data():
    if not os.path.exists(os.path.join(DATA_DIR, 'heart.csv')):
        df = pd.DataFrame(SAMPLE_DATA)
        df.to_csv(os.path.join(DATA_DIR, 'heart.csv'), index=False)
        return "Sample dataset created successfully"
    return "Dataset already exists"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    status = "healthy"
    warnings = []
    
    if not SKLEARN_AVAILABLE:
        warnings.append("scikit-learn not available")
        status = "degraded"
    if not JOBLIB_AVAILABLE:
        warnings.append("joblib not available")
        status = "degraded"
    if not PLOTLY_AVAILABLE:
        warnings.append("plotly not available")
        status = "degraded"
    
    return jsonify({
        "status": status, 
        "message": "Heart Disease Prediction API is running",
        "warnings": warnings
    })

@app.route('/api/download-template')
def download_template():
    """Download the sample dataset template"""
    try:
        # Create a more comprehensive template
        template_data = {
            'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57, 45, 50, 55, 60, 65],
            'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0, 0, 1, 1, 2, 2, 0, 1, 2, 3, 0],
            'trestbps': [145, 130, 130, 120, 120, 110, 140, 120, 172, 150, 110, 125, 135, 150, 160],
            'chol': [233, 250, 204, 236, 354, 177, 288, 263, 199, 126, 180, 200, 220, 250, 300],
            'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            'restecg': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2],
            'thalach': [150, 187, 172, 178, 163, 165, 133, 173, 162, 173, 170, 160, 150, 140, 130],
            'exang': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 2.6, 4.0, 0.0, 0.5, 1.6, 0.5, 1.0, 1.5, 2.0, 3.0],
            'slope': [0, 0, 2, 2, 2, 2, 3, 1, 1, 2, 2, 1, 0, 0, 0],
            'ca': [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 3],
            'thal': [1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 1, 2, 2, 3, 3],
            'target': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        }
        
        df = pd.DataFrame(template_data)
        template_path = os.path.join(DATA_DIR, 'template.csv')
        df.to_csv(template_path, index=False)
        
        return send_file(template_path, as_attachment=True, download_name='heart_disease_template.csv')
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/initialize-data', methods=['POST'])
def initialize_data():
    try:
        message = initialize_sample_data()
        return jsonify({"success": True, "message": message})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if file and allowed_file(file.filename):
            # Read the CSV file
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return jsonify({"success": False, "error": f"Error reading CSV file: {str(e)}"})
            
            # Validate the dataset
            is_valid, message = validate_dataset(df)
            if not is_valid:
                return jsonify({"success": False, "error": message})
            
            # Preprocess the dataset
            df = preprocess_dataset(df)
            
            # Save the processed dataset
            filepath = os.path.join(DATA_DIR, 'heart.csv')
            df.to_csv(filepath, index=False)
            
            return jsonify({
                "success": True, 
                "message": f"Dataset uploaded and processed successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                "dataset_info": {
                    "shape": df.shape,
                    "target_distribution": df['target'].value_counts().to_dict(),
                    "columns": list(df.columns)
                }
            })
        else:
            return jsonify({"success": False, "error": "Please upload a valid CSV file"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/dataset-info')
def dataset_info():
    try:
        if not os.path.exists(os.path.join(DATA_DIR, 'heart.csv')):
            return jsonify({"success": False, "error": "No dataset found. Please upload or initialize a dataset first."})
        
        df = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
        
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df['target'].value_counts().to_dict(),
            "sample_data": df.head(10).to_dict('records'),
            "feature_stats": {
                "age": {"min": int(df['age'].min()), "max": int(df['age'].max()), "mean": round(df['age'].mean(), 1)},
                "trestbps": {"min": int(df['trestbps'].min()), "max": int(df['trestbps'].max()), "mean": round(df['trestbps'].mean(), 1)},
                "chol": {"min": int(df['chol'].min()), "max": int(df['chol'].max()), "mean": round(df['chol'].mean(), 1)},
                "thalach": {"min": int(df['thalach'].min()), "max": int(df['thalach'].max()), "mean": round(df['thalach'].mean(), 1)}
            }
        }
        
        return jsonify({"success": True, "data": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/train-model', methods=['POST'])
def train_model():
    if not SKLEARN_AVAILABLE:
        return jsonify({"success": False, "error": "scikit-learn is not available. Please install it first."})
    
    try:
        data = request.json
        model_type = data.get('model_type', 'random_forest')
        
        # Check if dataset exists
        if not os.path.exists(os.path.join(DATA_DIR, 'heart.csv')):
            return jsonify({"success": False, "error": "No dataset found. Please upload or initialize a dataset first."})
        
        # Load data
        df = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
        
        # Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model based on type
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'naive_bayes':
            model = GaussianNB()
        elif model_type == 'svm':
            model = svm.SVC(kernel='linear', random_state=42, probability=True)
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=7)
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=100)
        elif model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            model = Sequential([
                Dense(64, activation='relu', input_dim=13),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2)
        else:
            return jsonify({"success": False, "error": f"Invalid model type or {model_type} not available"})
        
        # Train model (except neural network which is already trained)
        if model_type != 'neural_network':
            model.fit(X_train, y_train)
        
        # Make predictions
        if model_type == 'neural_network':
            y_pred = model.predict(X_test)
            y_pred = [round(x[0]) for x in y_pred]
        else:
            y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
        
        # Save model if joblib is available
        if JOBLIB_AVAILABLE:
            model_path = os.path.join(MODELS_DIR, f'{model_type}_model.pkl')
            if model_type == 'neural_network':
                model.save(model_path.replace('.pkl', '.h5'))
            else:
                joblib.dump(model, model_path)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        return jsonify({
            "success": True,
            "model_type": model_type,
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": feature_importance,
            "training_samples": len(X_train),
            "testing_samples": len(X_test)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not JOBLIB_AVAILABLE:
        return jsonify({"success": False, "error": "joblib is not available. Please install it first."})
    
    try:
        data = request.json
        model_type = data.get('model_type', 'random_forest')
        features = data.get('features', {})
        
        # Check if model exists
        model_path = os.path.join(MODELS_DIR, f'{model_type}_model.pkl')
        if model_type == 'neural_network':
            model_path = model_path.replace('.pkl', '.h5')
        
        if not os.path.exists(model_path):
            return jsonify({"success": False, "error": f"Model {model_type} not found. Please train the model first."})
        
        # Load model
        if model_type == 'neural_network':
            if TENSORFLOW_AVAILABLE:
                model = load_model(model_path)
            else:
                return jsonify({"success": False, "error": "TensorFlow not available for neural network"})
        else:
            model = joblib.load(model_path)
        
        # Prepare features
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        feature_values = [features.get(name, 0) for name in feature_names]
        
        # Make prediction
        if model_type == 'neural_network':
            prediction = model.predict([feature_values])
            probability = float(prediction[0][0])
            result = 1 if probability > 0.5 else 0
        else:
            prediction = model.predict([feature_values])
            result = int(prediction[0])
            
            # Get probability if available
            try:
                probability = float(model.predict_proba([feature_values])[0][1])
            except:
                probability = 0.5
        
        # Generate risk assessment
        risk_level = "HIGH" if result == 1 else "LOW"
        risk_description = "High Risk of Heart Disease" if result == 1 else "Low Risk of Heart Disease"
        
        # Generate recommendations based on features
        recommendations = []
        if features.get('age', 0) > 60:
            recommendations.append("Consider regular heart health checkups due to age")
        if features.get('trestbps', 0) > 140:
            recommendations.append("Monitor blood pressure regularly")
        if features.get('chol', 0) > 240:
            recommendations.append("Consider cholesterol management")
        if features.get('fbs', 0) == 1:
            recommendations.append("Monitor blood sugar levels")
        
        return jsonify({
            "success": True,
            "prediction": result,
            "probability": probability,
            "risk_level": risk_level,
            "result_text": risk_description,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/visualizations')
def get_visualizations():
    if not PLOTLY_AVAILABLE:
        return jsonify({"success": False, "error": "Plotly is not available. Please install it first."})
    
    try:
        if not os.path.exists(os.path.join(DATA_DIR, 'heart.csv')):
            return jsonify({"success": False, "error": "No dataset found. Please upload or initialize a dataset first."})
        
        df = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
        
        # Target distribution
        target_fig = px.pie(df, names='target', title='Target Distribution (0: No Disease, 1: Disease)',
                           color_discrete_map={0: '#2E8B57', 1: '#DC143C'})
        target_json = json.dumps(target_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Age distribution
        age_fig = px.histogram(df, x='age', title='Age Distribution', nbins=20,
                              color_discrete_sequence=['#4682B4'])
        age_json = json.dumps(age_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Correlation heatmap
        corr_matrix = df.corr()
        corr_fig = px.imshow(corr_matrix, title='Feature Correlation Heatmap', 
                            color_continuous_scale='RdBu', aspect='auto')
        corr_json = json.dumps(corr_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Sex vs Target
        sex_target_fig = px.bar(df.groupby(['sex', 'target']).size().reset_index(name='count'),
                               x='sex', y='count', color='target', 
                               title='Sex vs Target Distribution',
                               color_discrete_map={0: '#2E8B57', 1: '#DC143C'})
        sex_target_json = json.dumps(sex_target_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Feature distributions
        feature_fig = px.box(df, y=['age', 'trestbps', 'chol', 'thalach'], 
                           title='Feature Distributions')
        feature_json = json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            "success": True,
            "visualizations": {
                "target_distribution": target_json,
                "age_distribution": age_json,
                "correlation_heatmap": corr_json,
                "sex_target_distribution": sex_target_json,
                "feature_distributions": feature_json
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/models')
def get_models():
    try:
        models = []
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith(('.pkl', '.h5')):
                    model_name = filename.replace('_model.pkl', '').replace('_model.h5', '')
                    models.append({
                        "name": model_name.replace('_', ' ').title(),
                        "filename": filename,
                        "type": "Neural Network" if filename.endswith('.h5') else "Machine Learning"
                    })
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete-model', methods=['POST'])
def delete_model():
    try:
        data = request.json
        model_type = data.get('model_type')
        
        if not model_type:
            return jsonify({"success": False, "error": "Model type not specified"})
        
        model_path = os.path.join(MODELS_DIR, f'{model_type}_model.pkl')
        if model_type == 'neural_network':
            model_path = model_path.replace('.pkl', '.h5')
        
        if os.path.exists(model_path):
            os.remove(model_path)
            return jsonify({"success": True, "message": f"Model {model_type} deleted successfully"})
        else:
            return jsonify({"success": False, "error": f"Model {model_type} not found"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Initialize sample data on startup
    initialize_sample_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
