<<<<<<< HEAD
# CarDekho Used Car Price Prediction ML Model 🏎️💰

Predicts used car selling price based on vehicle_age, km_driven, brand, fuel_type, etc.

## Quick Start
```bash
pip install -r requirements.txt
python train_model.py  # Trains and saves model.pkl + plots
python prediction.py   # Test example prediction

## 🚀 Premium Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Features:**
- 🔮 Single prediction with impact analysis
- ⚖️ Compare 2 cars + AI recommendation (better value)
- 📈 Dataset insights & visualizations
- ⚙️ Live model retraining

**Pro Tip:** Set vehicle_age=0, km_driven=100 for new car prices!


```

## Files
- `data_preprocessing.py`: EDA, encoding, scaling, train/test split
- `model_training.py`: LinearRegression, RandomForest, XGBoost w/ GridSearchCV
- `model_evaluation.py`: RMSE/MAE/R2 metrics + plots
- `prediction.py`: Load model, predict new car price
- `train_model.py`: Full pipeline orchestrator

## Expected Performance
- Test RMSE: ~0.45-0.55 (log scale) → ~₹1-2L avg error
- R²: 0.85-0.92
- Feature importance: vehicle_age > km_driven > max_power > brand

## Prediction Example
```python
from prediction import predict_price
price = predict_price(vehicle_age=5, km_driven=50000, brand='Maruti', 
                      fuel_type='Petrol', transmission_type='Manual',
                      mileage=18.5, engine=1200, max_power=80, seats=5)
```

## Model Pipeline
1. Log-transform target (selling_price)
2. OneHotEncode: seller_type, fuel_type, transmission_type, brand
3. StandardScale: numerical features
4. XGBoost (auto-selected best)

**Data:** 15k+ CarDekho samples, no missing values.

Enjoy predicting car prices! 🚀
=======
# Car-Price-Prediction
>>>>>>> b030717167b02d18646c9062671b02ed53766129
