import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.pipeline import Pipeline
from features.feature_engineering import create_preprocessor

def get_regression_models():
    """Return a dictionary of regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    return models

def train_regression_model(X_train, X_test, y_train, y_test, preprocessor, model_name='Random Forest'):
    """Train a specific regression model and return the pipeline, predictions and metrics"""
    models = get_regression_models()
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', models[model_name])
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Store results
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape
    }
    
    return pipeline, y_pred, metrics

def evaluate_regression_models(df, cv=5):
    """Evaluate all regression models using cross-validation"""
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price']
    
    # Get preprocessor
    preprocessor = create_preprocessor()
    
    # Get regression models
    models = get_regression_models()
    
    # Results dictionary
    results = {}
    
    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Calculate cross-validation scores
        r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error'))
        mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        # Store results
        results[name] = {
            'R²': r2_scores.mean(),
            'R² Std': r2_scores.std(),
            'RMSE': rmse_scores.mean(),
            'RMSE Std': rmse_scores.std(),
            'MAE': mae_scores.mean(),
            'MAE Std': mae_scores.std(),
        }
    
    return pd.DataFrame(results).T
