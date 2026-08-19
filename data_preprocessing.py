import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class CarDataPreprocessor:
    def __init__(self):
        self.numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        self.categorical_features = ['seller_type', 'fuel_type', 'transmission_type', 'brand']
        self.target = 'selling_price'
        self.preprocessor = None
        self.label_encoders = {}
    
    def load_data(self, filepath):
        """Load and basic clean data"""
        df = pd.read_csv(filepath)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.drop('car_name', axis=1, inplace=True)  # Drop high cardinality, keep brand
        df.drop('model', axis=1, inplace=True)     # Drop high cardinality
        
        # Handle seats outliers (0 seats invalid)
        df = df[df['seats'] > 0]
        
        # Log transform skewed target
        df[self.target] = np.log1p(df[self.target])
        
        print(f"Data shape after cleaning: {df.shape}")
        print(df[self.numerical_features + self.categorical_features + [self.target]].describe())
        return df
    
    def prepare_features(self, df):
        """Prepare X and y"""
        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target]
        return X, y
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline"""
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        return self.preprocessor
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, X_train, X_test):
        """Fit and transform"""
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        return X_train_processed, X_test_processed
    
    def save_preprocessor(self, filepath='preprocessor.pkl'):
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")

if __name__ == "__main__":
    preprocessor = CarDataPreprocessor()
    df = preprocessor.load_data('cardekho_dataset.csv')
    X, y = preprocessor.prepare_features(df)
    preprocessor.create_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_proc, X_test_proc = preprocessor.fit_transform(X_train, X_test)
    preprocessor.save_preprocessor()
    print("Preprocessing complete!")
    print(f"X_train shape: {X_train_proc.shape}, y_train shape: {y_train.shape}")

