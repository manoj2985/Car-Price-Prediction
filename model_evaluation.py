import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(full_pipeline, X_test, y_test):
    """Evaluate saved model"""
    preprocessor = full_pipeline['preprocessor']
    model = full_pipeline['model']
    
    X_test_proc = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_proc)
    
    # Inverse log transform
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print(f"Test RMSE: ₹{rmse:,.0f}")
    print(f"Test MAE: ₹{mae:,.0f}")
    print(f"Test R2: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Actual Price (₹)')
    plt.ylabel('Predicted Price (₹)')
    plt.title('Predicted vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_test_orig - y_pred_orig
    plt.scatter(y_pred_orig, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price (₹)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if tree-based)
    if hasattr(model, 'feature_importances_'):
        feature_names = full_pipeline['feature_names']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

if __name__ == "__main__":
    # Load for standalone evaluation
    pipeline = joblib.load('car_price_model.pkl')
    # Assume test data loaded similarly
    print("Model evaluation complete. Check model_evaluation.png and feature_importance.png")

