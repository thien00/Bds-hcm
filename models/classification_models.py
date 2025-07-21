import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
from sklearn.pipeline import Pipeline
from features.feature_engineering import create_preprocessor

def get_classification_models():
    """Return a dictionary of classification models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    }
    return models

def train_classification_model(X_train, X_test, y_train, y_test, preprocessor, model_name='Random Forest'):
    """Train a specific classification model and return the pipeline, predictions and metrics"""
    models = get_classification_models()
    
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
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    metrics = {
        'Accuracy': accuracy,
        'Classification Report': report
    }
    
    return pipeline, y_pred, metrics

def evaluate_classification_models(df, cv=5):
    """Evaluate all classification models using cross-validation"""
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Get preprocessor
    preprocessor = create_preprocessor()
    
    # Get classification models
    models = get_classification_models()
    
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
        accuracy_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
        f1_macro_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro')
        precision_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='precision_macro')
        recall_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='recall_macro')
        
        # Store resultsa
        results[name] = {
            'Accuracy': accuracy_scores.mean(),
            'Accuracy Std': accuracy_scores.std(),
            'F1 Score': f1_macro_scores.mean(),
            'F1 Score Std': f1_macro_scores.std(),
            'Precision': precision_scores.mean(),
            'Precision Std': precision_scores.std(),
            'Recall': recall_scores.mean(),
            'Recall Std': recall_scores.std(),
        }
    
    return pd.DataFrame(results).T
