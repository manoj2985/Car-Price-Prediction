#!/usr/bin/env python3
"""
Main script to train the car price prediction model.
Run: python train_model.py
"""
from model_training import train_models
from data_preprocessing import CarDataPreprocessor
from model_evaluation import evaluate_model
import joblib

if __name__ == "__main__":
    print("🚗 Starting CarDekho Price Prediction Model Training...")
    
    # Train models
    full_pipeline, results = train_models()
    
    print("\n📊 Training completed!")
    print("Saved: car_price_model.pkl, preprocessor.pkl")
    
    # Quick evaluation
    preprocessor = CarDataPreprocessor()
    df = preprocessor.load_data('cardekho_dataset.csv')
    X, y = preprocessor.prepare_features(df)
    preprocessor.create_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_test_proc = preprocessor.preprocessor.transform(X_test)
    
    metrics = evaluate_model(full_pipeline, X_test, y_test)
    
    print("\n✅ Model ready for predictions!")
    print("Run: python prediction.py")
    print("\n📈 Results summary:")
    for model, scores in results.items():
        print(f"{model}: RMSE={scores['RMSE']:.4f}, R2={scores['R2']:.4f}")

