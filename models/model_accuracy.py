import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from data.data_preprocessing import load_and_preprocess_data, create_price_category
import plotly.express as px
import plotly.graph_objects as go

def get_preprocessor():
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

def evaluate_regression_models(df, cv=5):
    """Evaluate regression models with cross-validation"""
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price']
    
    # Preprocessor
    preprocessor = get_preprocessor()
    
    # Define regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
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

def evaluate_classification_models(df, cv=5):
    """Evaluate classification models with cross-validation"""
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Preprocessor
    preprocessor = get_preprocessor()
    
    # Define classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    }
    
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
        
        # Store results
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

def plot_regression_accuracy(reg_results):
    # Create a bar chart for R² scores
    fig = px.bar(
        reg_results.reset_index(), 
        x='index', 
        y='R²', 
        error_y='R² Std',
        title='Regression Models Performance (R²)',
        labels={'index': 'Model', 'R²': 'R² Score'},
        color='R²',
        color_continuous_scale='viridis'  # FIX: Changed Viridis to viridis
    )
    
    # Add a horizontal line for R²=0.7 as "good model" threshold
    fig.add_shape(type="line",
        x0=-0.5, y0=0.7, x1=3.5, y1=0.7,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def plot_classification_accuracy(cls_results):
    # Melt the DataFrame to create a format suitable for a grouped bar chart
    metrics_to_plot = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    melted_df = pd.melt(
        cls_results.reset_index(),
        id_vars='index',
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Score'
    )
    
    # Create a grouped bar chart
    fig = px.bar(
        melted_df,
        x='index',
        y='Score',
        color='Metric',
        title='Classification Models Performance Metrics',
        labels={'index': 'Model', 'Score': 'Score Value'},
        barmode='group'
    )
    
    # Add a horizontal line for accuracy=0.8 as "good model" threshold
    fig.add_shape(type="line",
        x0=-0.5, y0=0.8, x1=3.5, y1=0.8,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        legend_title="Metric"
    )
    
    return fig

def plot_class_specific_accuracy(df):
    """Plot accuracy by price category"""
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Preprocessor
    preprocessor = get_preprocessor()
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    }
    
    # Mapping for class labels
    category_labels = {
        0: 'Low (<1B)',
        1: 'Medium (1-3B)',
        2: 'High (3-7B)',
        3: 'Very High (>7B)'
    }
    
    # Results for each model by class
    all_results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name} by price category...")
        
        # Create and train pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy by class
        class_accuracy = {}
        for class_idx in np.unique(y):
            # Get indices where true label is this class
            idx = (y_test == class_idx)
            if np.any(idx):
                # Calculate accuracy for this class
                class_acc = accuracy_score(y_test[idx], y_pred[idx])
                class_accuracy[category_labels[class_idx]] = class_acc
        
        all_results[name] = class_accuracy
    
    # Convert to DataFrame
    class_acc_df = pd.DataFrame(all_results)
    
    # Create heatmap
    fig = px.imshow(
        class_acc_df,
        text_auto=True,
        labels=dict(x="Model", y="Price Category", color="Accuracy"),
        title="Model Accuracy by Price Category",
        color_continuous_scale="viridis"  # FIX: Changed Viridis to viridis
    )
    
    return fig

def generate_accuracy_report():
    """Generate a comprehensive accuracy report for all models"""
    # Load data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Evaluate models
    print("\nEvaluating regression models...")
    reg_results = evaluate_regression_models(df)
    
    print("\nEvaluating classification models...")
    cls_results = evaluate_classification_models(df)
    
    # Print results
    print("\n===== REGRESSION MODELS ACCURACY =====")
    print(reg_results[['R²', 'RMSE', 'MAE']])
    
    print("\n===== CLASSIFICATION MODELS ACCURACY =====")
    print(cls_results[['Accuracy', 'F1 Score', 'Precision', 'Recall']])
    
    # Return results for visualization
    return df, reg_results, cls_results

if __name__ == "__main__":
    # Generate report
    df, reg_results, cls_results = generate_accuracy_report()
    
    # Plot regression results
    plot_regression_accuracy(reg_results)
    plt.title("Regression Models Performance (R²)")
    plt.tight_layout()
    plt.savefig("regression_accuracy.png")
    
    # Plot classification results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar positions
    model_names = cls_results.index
    bar_width = 0.2
    bar_positions = np.arange(len(model_names))
    
    # Plot bars for different metrics
    ax.bar(bar_positions - 1.5*bar_width, cls_results['Accuracy'], bar_width, label='Accuracy')
    ax.bar(bar_positions - 0.5*bar_width, cls_results['F1 Score'], bar_width, label='F1 Score')
    ax.bar(bar_positions + 0.5*bar_width, cls_results['Precision'], bar_width, label='Precision')
    ax.bar(bar_positions + 1.5*bar_width, cls_results['Recall'], bar_width, label='Recall')
    
    # Customize plot
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Classification Models Performance')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("classification_accuracy.png")
    
    # Plot class-specific accuracy
    plot_class_specific_accuracy(df)
    plt.tight_layout()
    plt.savefig("class_specific_accuracy.png")
    
    print("\nAccuracy analysis complete. Visualization images saved.")
