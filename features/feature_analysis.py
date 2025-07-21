import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path='data/df.csv'):
    df = pd.read_csv(file_path)
    # Filter out rows with missing price
    df = df[df["Price"] != ""]
    # Ensure price is numeric
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
    # Filter out extreme prices
    df = df[df["Price"] <= 500].reset_index(drop=True)
    return df

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

def get_feature_importance(df):
    # Select features
    features = ['Acreage', 'Num_bedroom', 'Num_WC', 'District']
    X = df[features]
    y_reg = df['Price']
    y_cls = df['Price_Category'].astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Define numeric and categorical features
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
    
    # Train models to get feature importance
    rf_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    rf_cls = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit models
    rf_reg.fit(X, y_reg)
    rf_cls.fit(X, y_cls)
    
    # Get feature names after preprocessing
    feature_names = numeric_features.copy()
    # Add encoded categorical feature names
    encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    if hasattr(encoder, 'get_feature_names_out'):
        cat_feature_names = encoder.get_feature_names_out([categorical_features[0]])
    else:
        cat_feature_names = [f"{categorical_features[0]}_{cat}" for cat in encoder.categories_[0]]
    feature_names.extend(cat_feature_names)
    
    # Get feature importances
    reg_importances = rf_reg.named_steps['model'].feature_importances_
    cls_importances = rf_cls.named_steps['model'].feature_importances_
    
    return feature_names, reg_importances, cls_importances

def analyze_per_category(df):
    """Analyze important features for each price category"""
    categories = {
        0: 'Low price (< 1B)',
        1: 'Medium price (1-3B)', 
        2: 'High price (3-7B)',
        3: 'Very high price (> 7B)'
    }
    
    for category in categories.keys():
        print(f"\n=== Analysis for {categories[category]} ===")
        
        # Filter data for this category
        cat_df = df[df['Price_Category'] == category]
        
        # Basic statistics
        print(f"Count: {len(cat_df)}")
        print(f"Average Price: {cat_df['Price'].mean():.2f} billion")
        print(f"Average Acreage: {cat_df['Acreage'].mean():.2f} m²")
        print(f"Average Bedrooms: {cat_df['Num_bedroom'].mean():.2f}")
        print(f"Average Bathrooms: {cat_df['Num_WC'].mean():.2f}")
        
        # Top districts
        top_districts = cat_df['District'].value_counts().head(5)
        print("\nTop 5 Districts:")
        for district, count in top_districts.items():
            print(f"  {district}: {count} houses ({count/len(cat_df)*100:.1f}%)")
        
        # Price per m² analysis
        cat_df['Price_per_m2'] = cat_df['Price'] * 1e9 / cat_df['Acreage']  # Convert to VND/m²
        print(f"\nAverage Price per m²: {cat_df['Price_per_m2'].mean() / 1e6:.2f} million VND")
        
        print("-" * 50)

def plot_feature_importance(feature_names, reg_importances, cls_importances):
    # Sort features by regression importance
    reg_indices = np.argsort(reg_importances)[::-1]
    
    # Plot regression feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(reg_importances)), reg_importances[reg_indices])
    plt.xticks(range(len(reg_importances)), [feature_names[i] for i in reg_indices], rotation=90)
    plt.title('Feature Importance for Price Regression')
    plt.tight_layout()
    plt.show()
    
    # Sort features by classification importance
    cls_indices = np.argsort(cls_importances)[::-1]
    
    # Plot classification feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(cls_importances)), cls_importances[cls_indices])
    plt.xticks(range(len(cls_importances)), [feature_names[i] for i in cls_indices], rotation=90)
    plt.title('Feature Importance for Price Category Classification')
    plt.tight_layout()
    plt.show()
    
    # Compare top 10 features for both models
    top_n = 10
    top_reg_features = [feature_names[i] for i in reg_indices[:top_n]]
    top_cls_features = [feature_names[i] for i in cls_indices[:top_n]]
    
    print("\n=== Top 10 Features for Regression ===")
    for i, feature in enumerate(top_reg_features):
        print(f"{i+1}. {feature}: {reg_importances[reg_indices[i]]:.4f}")
    
    print("\n=== Top 10 Features for Classification ===")
    for i, feature in enumerate(top_cls_features):
        print(f"{i+1}. {feature}: {cls_importances[cls_indices[i]]:.4f}")
        
    # Features in common
    common_features = set(top_reg_features) & set(top_cls_features)
    print(f"\nFeatures in common: {len(common_features)} out of {top_n}")
    for feature in common_features:
        print(f"- {feature}")

def create_price_category_correlation_heatmap(df):
    # Select relevant numeric features
    numeric_df = df[['Price', 'Price_Category', 'Acreage', 'Num_bedroom', 'Num_WC']].copy()
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Features with Price and Price Category')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Create price category
    df = create_price_category(df)
    
    # Basic analysis of each price category
    print("\n=== Analysis by Price Category ===")
    analyze_per_category(df)
    
    # Calculate feature importance
    print("\n=== Feature Importance Analysis ===")
    feature_names, reg_importances, cls_importances = get_feature_importance(df)
    plot_feature_importance(feature_names, reg_importances, cls_importances)
    
    # Create correlation heatmap
    print("\n=== Correlation Analysis ===")
    create_price_category_correlation_heatmap(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
