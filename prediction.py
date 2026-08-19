import joblib
import pandas as pd
import numpy as np

def predict_price(model_path='car_price_model.pkl', **kwargs):
    """
    Predict car price. Example:
    predict_price(
        vehicle_age=5, km_driven=50000, seller_type='Individual',
        fuel_type='Petrol', transmission_type='Manual', mileage=18.5,
        engine=1200, max_power=80, seats=5, brand='Maruti'
    )
    """
    pipeline = joblib.load(model_path)
    preprocessor = pipeline['preprocessor']
    
    # Create input DataFrame
    input_data = pd.DataFrame([kwargs])
    
    # Preprocess
    input_processed = preprocessor.transform(input_data)
    
    # Predict (log scale)
    log_pred = pipeline['model'].predict(input_processed)[0]
    
    # Inverse transform
    pred_price = np.expm1(log_pred)
    
    print(f"Predicted selling price: ₹{pred_price:,.0f}")
    return pred_price

if __name__ == "__main__":
    # Example prediction
    price = predict_price(
        vehicle_age=5,
        km_driven=50000,
        seller_type='Individual',
        fuel_type='Petrol',
        transmission_type='Manual',
        mileage=18.5,
        engine=1200,
        max_power=80,
        seats=5,
        brand='Maruti'
    )

