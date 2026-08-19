# CarDekho Used Car Price Prediction

A machine-learning project that estimates the selling price of a used car from its specifications. It includes a model-training pipeline, a Python prediction helper, and an interactive Streamlit dashboard for predictions, comparisons, and dataset exploration.

## Features

- Train and compare Linear Regression, Random Forest, and XGBoost models.
- Preprocess numeric features with standard scaling.
- Encode categorical features with one-hot encoding.
- Predict prices after applying the inverse of the target's  log1p  transformation.
- Explore the dataset and compare two cars in the Streamlit app.

## Requirements

- Python 3.9 or newer
- The included  cardekho_dataset.csv  file

Install the dependencies in a virtual environment:

   bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
   

## Quick Start

### 1. Train the model

Run the training pipeline from the project directory:

   bash
python train_model.py
   

This cleans the dataset, trains the candidate models with cross-validated parameter search, evaluates them, selects the model with the lowest RMSE, and writes:

   text
car_price_model.pkl
   

The generated model file is required for predictions and for the Streamlit dashboard. It is intentionally not committed to source control because it can be regenerated from the CSV dataset.

### 2. Try a command-line prediction

   bash
python prediction.py
   

Or use the helper in another Python script:

   python
from prediction import predict_price

price = predict_price(
    vehicle_age=5,
    km_driven=50000,
    seller_type="Individual",
    fuel_type="Petrol",
    transmission_type="Manual",
    mileage=18.5,
    engine=1200,
    max_power=80,
    seats=5,
    brand="Maruti",
)

print(f"Estimated price: ₹{price:,.0f}")
   

### 3. Launch the dashboard

Train the model first, then start Streamlit:

   bash
streamlit run app.py
   

The dashboard provides single-car prediction, two-car comparison, visual dataset insights, and additional car-suggestion functionality.

## Model Inputs

The prediction API expects these fields:

| Type | Features |
| --- | --- |
| Numeric |  vehicle_age ,  km_driven ,  mileage ,  engine ,  max_power ,  seats  |
| Categorical |  seller_type ,  fuel_type ,  transmission_type ,  brand  |

The target column is  selling_price . The preprocessing step removes the dataset index,  car_name , and  model , filters invalid seat counts, and applies  log1p  to the target before training.

## Project Structure

| File | Purpose |
| --- | --- |
|  app.py  | Streamlit dashboard and interactive visualizations |
|  data_preprocessing.py  | Dataset cleaning, feature selection, encoding, scaling, and splitting |
|  model_training.py  | Model training, grid search, evaluation, and artifact creation |
|  model_evaluation.py  | Regression metrics and evaluation plots |
|  prediction.py  | Load the trained artifact and estimate a car price |
|  train_model.py  | End-to-end training entry point |
|  cardekho_dataset.csv  | Training and exploration dataset |

## Notes

- Run commands from the repository root so the scripts can find the CSV and generated model file.
- Model quality depends on the dataset and the selected features; retrain after changing the data.
- Predictions are estimates for analysis and should not be treated as a vehicle appraisal or purchase guarantee.
