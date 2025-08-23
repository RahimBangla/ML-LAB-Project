# Heart Disease Prediction System

A comprehensive web application for predicting heart disease risk using multiple machine learning algorithms. This system provides a user-friendly interface for data management, model training, prediction, and data visualization.

## Features

### 🏥 **Heart Disease Prediction**

- **Multiple ML Models**: Support for 8 different algorithms including Random Forest, Neural Networks, XGBoost, and more
- **Real-time Prediction**: Input patient parameters and get instant risk assessment
- **Confidence Scoring**: Each prediction includes probability scores and confidence levels

### 📊 **Data Management**

- **Sample Dataset**: Built-in heart disease dataset for immediate testing
- **CSV Upload**: Support for custom dataset uploads
- **Data Validation**: Automatic data quality checks and preprocessing

### 🤖 **Model Training**

- **Automated Training**: One-click model training with automatic hyperparameter optimization
- **Performance Metrics**: Accuracy scores, confusion matrices, and classification reports
- **Model Persistence**: Trained models are automatically saved for future use

### 📈 **Data Analytics & Visualization**

- **Interactive Charts**: Target distribution, age analysis, correlation heatmaps
- **Statistical Insights**: Comprehensive dataset statistics and feature analysis
- **Real-time Updates**: Dynamic charts that update with new data

## Supported Machine Learning Models

1. **Random Forest** - High accuracy, good for complex patterns
2. **Logistic Regression** - Fast, interpretable results
3. **Neural Network** - Deep learning approach with high performance
4. **Support Vector Machine (SVM)** - Good for high-dimensional data
5. **K-Nearest Neighbors (KNN)** - Simple, effective classification
6. **Decision Tree** - Interpretable decision paths
7. **Naive Bayes** - Probabilistic classification
8. **XGBoost** - Gradient boosting with excellent performance

## Dataset Features

The system uses the standard heart disease dataset with the following parameters:

- **Age**: Patient age in years
- **Sex**: Gender (1 = Male, 0 = Female)
- **Chest Pain Type**: 4 categories from typical angina to asymptomatic
- **Resting Blood Pressure**: Systolic pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: > 120 mg/dl (1 = Yes, 0 = No)
- **Resting ECG**: Results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)
- **Maximum Heart Rate**: Achieved during exercise
- **Exercise Induced Angina**: 1 = Yes, 0 = No
- **ST Depression**: Induced by exercise relative to rest
- **Slope**: Peak exercise ST segment slope
- **Number of Major Vessels**: Colored by fluoroscopy (0-3)
- **Thalassemia**: 3 = Normal, 6 = Fixed defect, 7 = Reversible defect

## Data Management & CSV Upload

### 📁 **CSV Dataset Upload**

The system now supports uploading your own heart disease datasets in CSV format. This allows you to:

- **Use Your Own Data**: Train models on datasets specific to your research or clinical needs
- **Custom Datasets**: Upload datasets with different patient populations or additional features
- **Real-world Data**: Use actual clinical data for more accurate predictions
- **Data Validation**: Automatic validation ensures your data meets the required format

### 📋 **Required Dataset Format**

Your CSV file must contain the following columns in this exact order:

| Column     | Description                       | Values               | Example |
| ---------- | --------------------------------- | -------------------- | ------- |
| `age`      | Patient age in years              | 1-120                | 63      |
| `sex`      | Gender                            | 0 (Female), 1 (Male) | 1       |
| `cp`       | Chest pain type                   | 0-3                  | 3       |
| `trestbps` | Resting blood pressure (mm Hg)    | 90-200               | 145     |
| `chol`     | Serum cholesterol (mg/dl)         | 100-600              | 233     |
| `fbs`      | Fasting blood sugar > 120 mg/dl   | 0 (No), 1 (Yes)      | 1       |
| `restecg`  | Resting ECG results               | 0-2                  | 0       |
| `thalach`  | Maximum heart rate achieved       | 60-202               | 150     |
| `exang`    | Exercise induced angina           | 0 (No), 1 (Yes)      | 0       |
| `oldpeak`  | ST depression induced by exercise | 0.0-6.2              | 2.3     |
| `slope`    | Peak exercise ST segment slope    | 0-2                  | 0       |
| `ca`       | Number of major vessels (0-3)     | 0-3                  | 0       |
| `thal`     | Thalassemia                       | 1-3                  | 1       |
| `target`   | Heart disease presence            | 0 (No), 1 (Yes)      | 1       |

### 🔍 **Data Validation**

The system automatically validates uploaded datasets:

- **Column Check**: Ensures all required columns are present
- **Data Types**: Verifies numeric columns contain valid numbers
- **Target Values**: Confirms target column contains only 0 and 1
- **Missing Values**: Checks for and handles any missing data
- **Data Preprocessing**: Automatically cleans and prepares data for training

### 📥 **How to Upload Your Dataset**

1. **Prepare Your Data**: Ensure your CSV file follows the required format
2. **Upload File**: Click the upload area in the Data Management tab
3. **Select File**: Choose your CSV file from your computer
4. **Automatic Processing**: The system validates and processes your data
5. **Ready to Use**: Your dataset is now available for model training

### 📊 **Download Template**

Not sure about the format? Download our template:

1. Go to the **Data Management** tab
2. Click **"Download Template"**
3. Use the template as a reference for your own dataset
4. Fill in your data following the same structure

### 💡 **Dataset Tips**

- **Data Quality**: Ensure your data is clean and well-formatted
- **Sample Size**: Larger datasets generally provide better model performance
- **Balance**: Try to have a balanced distribution of disease/no-disease cases
- **Missing Values**: Handle missing values before uploading (or let the system handle them)
- **Feature Scaling**: The system automatically handles feature scaling for most algorithms

### 🔄 **Switching Between Datasets**

You can easily switch between different datasets:

1. **Upload New Dataset**: Upload a new CSV file to replace the current one
2. **Initialize Sample Data**: Return to the built-in sample dataset
3. **Model Compatibility**: Note that models trained on one dataset may not work optimally on another

### 📈 **Dataset Analytics**

After uploading, view comprehensive dataset information:

- **Basic Statistics**: Row count, column count, data types
- **Target Distribution**: Balance between disease/no-disease cases
- **Feature Statistics**: Min, max, and average values for key features
- **Data Quality**: Missing value counts and data type information
- **Sample Records**: Preview of your actual data

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Option 1: Automatic Installation (Recommended)

```bash
# Run the automatic installation script
python install.py
```

### Option 2: Manual Installation

```bash
# Step 1: Clone the Repository
git clone <repository-url>
cd ML-LAB-Project

# Step 2: Install Basic Dependencies
pip install --upgrade setuptools wheel
pip install Flask Flask-CORS pandas numpy requests

# Step 3: Install ML Dependencies
pip install scikit-learn joblib plotly

# Step 4: Install Optional Dependencies (if needed)
pip install xgboost tensorflow matplotlib seaborn
```

### Option 3: Simple Requirements (Minimal Setup)

```bash
# Install only essential packages
pip install -r requirements-simple.txt
```

## Running the Application

### Method 1: Using the Startup Script (Recommended)

```bash
python run.py --init-data
```

### Method 2: Direct Execution

```bash
python app.py
```

### Method 3: Windows Batch File

```bash
start.bat
```

The application will be available at `http://localhost:5000`

## Usage Guide

### 1. **Dashboard**

- View system overview and statistics
- Monitor trained models and their performance
- Quick access to all system features

### 2. **Data Management**

- **Initialize Sample Data**: Load the built-in heart disease dataset
- **Upload Custom Data**: Upload your own CSV file with the same column structure
- **View Dataset Info**: Check data shape, missing values, and sample records

### 3. **Model Training**

- Select a machine learning algorithm from the dropdown
- Click "Train Model" to start training
- View training results including accuracy and confusion matrix
- Models are automatically saved for future predictions

### 4. **Prediction**

- Fill in patient parameters (age, sex, blood pressure, etc.)
- Select a trained model for prediction
- Get instant risk assessment with confidence scores
- Results show whether the patient is at high or low risk

### 5. **Analytics & Visualization**

- **Target Distribution**: Pie chart showing disease vs. no disease cases
- **Age Distribution**: Histogram of patient ages
- **Correlation Heatmap**: Feature relationships and importance
- **Sex vs. Target**: Gender-based disease distribution

## API Endpoints

The system provides a RESTful API for programmatic access:

- `GET /api/health` - System health check
- `POST /api/initialize-data` - Initialize sample dataset
- `POST /api/upload-data` - Upload custom CSV file
- `GET /api/dataset-info` - Get dataset statistics
- `POST /api/train-model` - Train a new ML model
- `POST /api/predict` - Make heart disease predictions
- `GET /api/visualizations` - Get data visualization charts
- `GET /api/models` - List available trained models

## Model Performance

Based on the original analysis, typical performance metrics:

- **Random Forest**: 85-90% accuracy
- **Neural Network**: 80-85% accuracy
- **XGBoost**: 85-88% accuracy
- **Logistic Regression**: 75-80% accuracy
- **SVM**: 80-85% accuracy

## Technical Architecture

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Machine Learning**: Scikit-learn, TensorFlow, XGBoost
- **Data Visualization**: Plotly.js
- **Data Processing**: Pandas, NumPy
- **Model Storage**: Joblib for ML models, HDF5 for neural networks

## File Structure

```
ML-LAB-Project/
├── app.py                 # Main Flask application
├── install.py             # Automatic installation script
├── requirements.txt       # Full Python dependencies
├── requirements-simple.txt # Minimal dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   └── index.html       # Main application interface
├── data/                # Dataset storage
├── models/              # Trained model storage
└── uploads/             # File upload temporary storage
```

## Troubleshooting

### Common Installation Issues

1. **setuptools.build_meta Error**

   ```bash
   pip install --upgrade setuptools wheel
   pip install --upgrade pip
   ```

2. **Build Tools Missing (Windows)**

   ```bash
   # Install Visual C++ Build Tools
   pip install --upgrade setuptools wheel
   ```

3. **Package Version Conflicts**

   ```bash
   # Use the simple requirements file
   pip install -r requirements-simple.txt
   ```

4. **Python Version Issues**
   - Ensure Python 3.8+ is installed
   - Use `python --version` to check

### Runtime Issues

1. **Port Already in Use**

   ```bash
   # Change port in app.py or run.py
   python run.py --port 5001
   ```

2. **Missing Dependencies**

   ```bash
   # Run the installation script
   python install.py
   ```

3. **Model Training Errors**
   - Check if dataset is loaded
   - Verify all required packages are installed
   - Check console for specific error messages

## Testing the System

### Run the Demo

```bash
python demo.py
```

### Test the API

```bash
python test_system.py
```

### Manual Testing

1. Start the application
2. Open http://localhost:5000
3. Navigate through all tabs
4. Try training a model
5. Make a prediction

## Performance Optimization

- **Model Caching**: Trained models are automatically saved and reused
- **Data Preprocessing**: Automatic handling of missing values and data types
- **Efficient Algorithms**: Optimized implementations for fast training and prediction
- **Memory Management**: Efficient data handling for large datasets

## Security Considerations

- Input validation for all user inputs
- File upload restrictions (CSV only)
- No sensitive data storage
- Local deployment for data privacy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the API documentation
3. Run the installation script: `python install.py`
4. Open an issue on the repository

## Future Enhancements

- **Real-time Data Streaming**: Live data ingestion and model updates
- **Advanced Analytics**: More sophisticated statistical analysis
- **Mobile App**: Native mobile application
- **Cloud Deployment**: AWS/Azure integration
- **Multi-language Support**: Internationalization
- **Advanced ML**: Deep learning models and ensemble methods

---

**Note**: This system is designed for educational and research purposes. For medical applications, please consult healthcare professionals and ensure compliance with relevant regulations.
