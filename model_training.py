import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from data_preprocessing import CarDataPreprocessor

def train_models():
    """Train multiple models with hyperparam tuning"""
    preprocessor = CarDataPreprocessor()
    df = preprocessor.load_data('cardekho_dataset.csv')
    X, y = preprocessor.prepare_features(df)
    preprocessor.create_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_proc, X_test_proc = preprocessor.fit_transform(X_train, X_test)
    
    models = {}
    
    # 1. Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train_proc, y_train)
    models['LinearRegression'] = lr
    
    # 2. Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_squared_error')
    rf.fit(X_train_proc, y_train)
    models['RandomForest'] = rf.best_estimator_
    print(f"Best RF params: {rf.best_params_}")
    
    # 3. XGBoost (likely best)
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=3, scoring='neg_mean_squared_error')
    xgb_model.fit(X_train_proc, y_train)
    models['XGBoost'] = xgb_model.best_estimator_
    print(f"Best XGB params: {xgb_model.best_params_}")
    
    # Evaluate all
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_proc)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    # Select best model (lowest RMSE)
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name}")
    
    # Save best model pipeline
    full_pipeline = {
        'preprocessor': preprocessor.preprocessor,
        'model': best_model,
        'feature_names': preprocessor.numerical_features + preprocessor.categorical_features,
        'target_transform': 'log1p'  # Note inverse needed for predictions
    }
    joblib.dump(full_pipeline, 'car_price_model.pkl')
    print("Best model saved as car_price_model.pkl")
    
    return full_pipeline, results

if __name__ == "__main__":
    pipeline, results = train_models()

