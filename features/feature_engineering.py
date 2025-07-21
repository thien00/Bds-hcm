import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def create_price_category(df):
    """
    Create price categories:
    - Tag 0: Low price (< 1 billion)
    - Tag 1: Medium price (1-3 billion)
    - Tag 2: High price (3-7 billion)
    - Tag 3: Very high price (> 7 billion)
    """
    conditions = [
        (df['Price'] < 1),
        (df['Price'] >= 1) & (df['Price'] < 3),
        (df['Price'] >= 3) & (df['Price'] < 7),
        (df['Price'] >= 7)
    ]
    choices = [0, 1, 2, 3]
    df['Price_Category'] = np.select(conditions, choices, default=np.nan)
    return df

def create_preprocessor():
    """Create the standard preprocessor for all models"""
    numeric_features = ['Acreage', 'Num_bedroom', 'Num_WC']
    categorical_features = ['District']
    
    # Create preprocessors
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_train_test_data(df):
    """Prepare train and test datasets for model training"""
    # Select features and target
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y_reg = df['Price']
    y_cls = df['Price_Category'].astype(int)
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test

def extract_feature_importances(model, preprocessor):
    """Extract feature importances with proper feature names"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get feature names
    numeric_features = ['Acreage', 'Num_bedroom', 'Num_WC']
    try:
        categorical_features = preprocessor.transformers_[1][2]
        encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        if hasattr(encoder, 'get_feature_names_out'):
            encoded_features = list(encoder.get_feature_names_out(categorical_features))
        else:
            encoded_features = [f"District_{cat}" for cat in encoder.categories_[0]]
        feature_names = numeric_features + encoded_features
    except:
        feature_names = list(range(len(model.feature_importances_)))
    
    # Create feature importance DataFrame
    importances_df = pd.DataFrame({
        'Feature': feature_names[:len(model.feature_importances_)],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importances_df
