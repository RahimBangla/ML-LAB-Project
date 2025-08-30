import os

class Config:
    """Configuration class for the Heart Disease Prediction System"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Application Settings
    APP_NAME = 'Heart Disease Prediction System'
    APP_VERSION = '1.0.0'
    
    # Directory Configuration
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    
    # File Configuration
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model Configuration
    DEFAULT_MODEL = 'random_forest'
    SUPPORTED_MODELS = [
        'logistic_regression',
        'naive_bayes',
        'svm',
        'knn',
        'decision_tree',
        'random_forest',
        'xgboost',
        'neural_network'
    ]
    
    # Training Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    NEURAL_NETWORK_EPOCHS = 100
    
    # Feature Configuration
    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Feature Descriptions
    FEATURE_DESCRIPTIONS = {
        'age': 'Patient age in years',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1-3)'
    }
    
    # Feature Ranges (for validation)
    FEATURE_RANGES = {
        'age': (1, 120),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (90, 200),
        'chol': (100, 600),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (60, 202),
        'exang': (0, 1),
        'oldpeak': (0.0, 6.2),
        'slope': (0, 2),
        'ca': (0, 3),
        'thal': (1, 3)
    }
    
    # API Configuration
    API_PREFIX = '/api'
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def init_app(cls, app):
        """Initialize the application with configuration"""
        # Create necessary directories
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.UPLOAD_FOLDER]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Set Flask configuration
        app.config['SECRET_KEY'] = cls.SECRET_KEY
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = cls.UPLOAD_FOLDER

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production-specific settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
