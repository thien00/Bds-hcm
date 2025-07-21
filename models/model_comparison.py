import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score
import xgboost as xgb

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

def get_preprocessor():
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
    
    return preprocessor

def compare_regression_models(df):
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price']
    
    # Get preprocessor
    preprocessor = get_preprocessor()
    
    # Define regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    # Store results
    results = {}
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Calculate cross-validation scores
        r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
        mse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
        
        # Store results
        results[name] = {
            'R²': r2_scores.mean(),
            'R² Std': r2_scores.std(),
            'MAE': mae_scores.mean(),
            'MAE Std': mae_scores.std(),
            'MSE': mse_scores.mean(),
            'MSE Std': mse_scores.std(),
            'RMSE': np.sqrt(mse_scores.mean())
        }
    
    # Convert to DataFrame for easier visualization
    results_df = pd.DataFrame(results).T
    
    return results_df

def compare_classification_models(df):
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Get preprocessor
    preprocessor = get_preprocessor()
    
    # Define classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    }
    
    # Store results
    results = {}
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Calculate cross-validation scores
        accuracy_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
        
        # Custom function for micro f1 score
        def micro_f1_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return f1_score(y, y_pred, average='micro')
        
        def macro_f1_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return f1_score(y, y_pred, average='macro')
        
        micro_f1_scores = np.array([micro_f1_scorer(pipeline.fit(X.iloc[train], y.iloc[train]), X.iloc[test], y.iloc[test]) 
                                   for train, test in kf.split(X)])
        
        macro_f1_scores = np.array([macro_f1_scorer(pipeline.fit(X.iloc[train], y.iloc[train]), X.iloc[test], y.iloc[test]) 
                                   for train, test in kf.split(X)])
        
        # Store results
        results[name] = {
            'Accuracy': accuracy_scores.mean(),
            'Accuracy Std': accuracy_scores.std(),
            'Micro F1': micro_f1_scores.mean(),
            'Micro F1 Std': micro_f1_scores.std(),
            'Macro F1': macro_f1_scores.mean(),
            'Macro F1 Std': macro_f1_scores.std()
        }
    
    # Convert to DataFrame for easier visualization
    results_df = pd.DataFrame(results).T
    
    return results_df

def plot_model_comparison(reg_results_df, cls_results_df):
    # Plot regression model comparison
    plt.figure(figsize=(12, 6))
    ax = reg_results_df[['R²', 'RMSE']].plot(kind='bar', secondary_y=['RMSE'], alpha=0.7)
    ax.set_ylabel('R² Score')
    ax.right_ax.set_ylabel('RMSE')
    plt.title('Regression Models Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot classification model comparison
    plt.figure(figsize=(12, 6))
    cls_results_df[['Accuracy', 'Macro F1']].plot(kind='bar', alpha=0.7)
    plt.title('Classification Models Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

def analyze_misclassifications(df):
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get preprocessor
    preprocessor = get_preprocessor()
    
    # Best classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create and train pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Find misclassified samples
    misclassified = X_test.copy()
    misclassified['Actual_Category'] = y_test.values
    misclassified['Predicted_Category'] = y_pred
    misclassified['Price'] = df.loc[X_test.index, 'Price'].values
    misclassified = misclassified[misclassified['Actual_Category'] != misclassified['Predicted_Category']]
    
    # Add category labels
    category_labels = {
        0: 'Low price (< 1B)',
        1: 'Medium price (1-3B)', 
        2: 'High price (3-7B)',
        3: 'Very high price (> 7B)'
    }
    misclassified['Actual_Label'] = misclassified['Actual_Category'].map(category_labels)
    misclassified['Predicted_Label'] = misclassified['Predicted_Category'].map(category_labels)
    
    # Analyze misclassifications
    print(f"\nTotal misclassified samples: {len(misclassified)} out of {len(X_test)} ({len(misclassified)/len(X_test)*100:.2f}%)")
    
    # Count misclassifications by category
    misclass_by_actual = misclassified['Actual_Category'].value_counts().sort_index()
    total_by_actual = y_test.value_counts().sort_index()
    
    print("\nMisclassification rate by actual category:")
    for category in sorted(y_test.unique()):
        if category in misclass_by_actual:
            rate = misclass_by_actual[category] / total_by_actual[category] * 100
            print(f"  {category_labels[category]}: {rate:.2f}% ({misclass_by_actual[category]} out of {total_by_actual[category]})")
        else:
            print(f"  {category_labels[category]}: 0.00% (0 out of {total_by_actual[category]})")
    
    # Plot misclassification matrix
    conf_mat = pd.crosstab(
        misclassified['Actual_Category'], 
        misclassified['Predicted_Category'], 
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Misclassification Matrix')
    plt.show()
    
    # Analyze common misclassification patterns
    print("\nCommon misclassification patterns:")
    error_patterns = misclassified.groupby(['Actual_Label', 'Predicted_Label']).size().sort_values(ascending=False)
    for (actual, predicted), count in error_patterns.items():
        print(f"  {actual} misclassified as {predicted}: {count} instances")
    
    return misclassified

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_data()
    df = create_price_category(df)
    
    # Compare regression models
    print("\n=== Comparing Regression Models ===")
    reg_results_df = compare_regression_models(df)
    print(reg_results_df)
    
    # Compare classification models
    print("\n=== Comparing Classification Models ===")
    cls_results_df = compare_classification_models(df)
    print(cls_results_df)
    
    # Plot model comparisons
    print("\nPlotting model comparisons...")
    plot_model_comparison(reg_results_df, cls_results_df)
    
    # Analyze misclassifications
    print("\n=== Analyzing Misclassifications ===")
    misclassified = analyze_misclassifications(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
