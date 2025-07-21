import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

# Import from new module files
from data.data_preprocessing import load_and_preprocess_data
from features.feature_engineering import create_price_category, create_preprocessor, prepare_train_test_data, extract_feature_importances
from models.regression_models import train_regression_model, get_regression_models, evaluate_regression_models
from models.classification_models import train_classification_model, get_classification_models, evaluate_classification_models
from models.model_evaluation import plot_regression_results, plot_confusion_matrix, plot_classification_accuracy_by_class

# Load the data
def load_data(file_path='data/df.csv'):
    # Use the comprehensive preprocessing function
    return load_and_preprocess_data(file_path)

def analyze_price_segments(df):
    """Analyze factors influencing each price segment"""
    price_categories = {
        0: 'Low price (< 1B)',
        1: 'Medium price (1-3B)',
        2: 'High price (3-7B)',
        3: 'Very high price (> 7B)'
    }
    
    # Distribution of houses by price category
    plt.figure(figsize=(10, 6))
    df['Price_Category'].value_counts().sort_index().plot(kind='bar')
    plt.xticks(ticks=range(4), labels=[price_categories[i] for i in range(4)])
    plt.title('Distribution of Houses by Price Category')
    plt.xlabel('Price Category')
    plt.ylabel('Count')
    plt.show()
    
    # Average area by price category
    plt.figure(figsize=(10, 6))
    df.groupby('Price_Category')['Acreage'].mean().plot(kind='bar')
    plt.xticks(ticks=range(4), labels=[price_categories[i] for i in range(4)])
    plt.title('Average Acreage by Price Category')
    plt.xlabel('Price Category')
    plt.ylabel('Average Acreage (m²)')
    plt.show()
    
    # Average number of bedrooms and bathrooms by price category
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    df.groupby('Price_Category')['Num_bedroom'].mean().plot(kind='bar', ax=ax[0])
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([price_categories[i] for i in range(4)])
    ax[0].set_title('Average Number of Bedrooms by Price Category')
    ax[0].set_xlabel('Price Category')
    ax[0].set_ylabel('Average Number of Bedrooms')
    
    df.groupby('Price_Category')['Num_WC'].mean().plot(kind='bar', ax=ax[1])
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([price_categories[i] for i in range(4)])
    ax[1].set_title('Average Number of Bathrooms by Price Category')
    ax[1].set_xlabel('Price Category')
    ax[1].set_ylabel('Average Number of Bathrooms')
    plt.tight_layout()
    plt.show()
    
    # District distribution by price category
    top_districts = df['District'].value_counts().nlargest(10).index
    plt.figure(figsize=(12, 8))
    for i, district in enumerate(top_districts):
        district_data = df[df['District'] == district]['Price_Category'].value_counts(normalize=True).sort_index() * 100
        plt.bar(np.arange(4) + i*0.1, district_data, width=0.1, label=district)
    
    plt.xticks(ticks=range(4), labels=[price_categories[i] for i in range(4)])
    plt.title('District Distribution by Price Category')
    plt.xlabel('Price Category')
    plt.ylabel('Percentage (%)')
    plt.legend(title='District')
    plt.show()

def compare_approaches(reg_results, cls_results):
    """Compare regression and classification approaches"""
    print("\n=== Comparison of Regression and Classification Approaches ===")
    
    # Print regression results
    print("\nRegression Models Performance:")
    for model, metrics in reg_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Print classification results
    print("\nClassification Models Performance:")
    for model, metrics in cls_results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        
        # Print average precision, recall, and f1-score
        avg_precision = np.mean([metrics['Classification Report'][str(i)]['precision'] for i in range(4)])
        avg_recall = np.mean([metrics['Classification Report'][str(i)]['recall'] for i in range(4)])
        avg_f1 = np.mean([metrics['Classification Report'][str(i)]['f1-score'] for i in range(4)])
        
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1-Score: {avg_f1:.4f}")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_data()
    df = create_price_category(df)
    
    # Preprocess data for both regression and classification
    print("Preprocessing data...")
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = prepare_train_test_data(df)
    preprocessor = create_preprocessor()
    
    # Train and evaluate regression models
    print("Training regression models...")
    reg_results = {}
    reg_importances = {}
    
    for model_name in get_regression_models().keys():
        pipeline, y_pred, metrics = train_regression_model(
            X_train, X_test, y_reg_train, y_reg_test, preprocessor, model_name
        )
        reg_results[model_name] = metrics
        
        # Store feature importances if available
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances_df = extract_feature_importances(pipeline.named_steps['model'], preprocessor)
            if importances_df is not None:
                reg_importances[model_name] = importances_df
    
    # Train and evaluate classification models
    print("Training classification models...")
    cls_results = {}
    cls_importances = {}
    cls_predictions = {}
    
    for model_name in get_classification_models().keys():
        pipeline, y_pred, metrics = train_classification_model(
            X_train, X_test, y_cls_train, y_cls_test, preprocessor, model_name
        )
        cls_results[model_name] = metrics
        cls_predictions[model_name] = y_pred
        
        # Store feature importances if available
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances_df = extract_feature_importances(pipeline.named_steps['model'], preprocessor)
            if importances_df is not None:
                cls_importances[model_name] = importances_df
    
    # Compare approaches
    compare_approaches(reg_results, cls_results)
    
    # Plot confusion matrix for the best classification model
    print("\nGenerating confusion matrix for Random Forest Classifier...")
    class_names = ['Low price (<1B)', 'Medium price (1-3B)', 'High price (3-7B)', 'Very high price (>7B)']
    fig = plot_confusion_matrix(y_cls_test, cls_predictions['Random Forest'], class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_cls_test, cls_predictions['Random Forest']), 
                annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Analyze price segments
    print("Analyzing price segments...")
    analyze_price_segments(df)
    
    # Cross-validation evaluation of models
    print("\nEvaluating models with cross-validation...")
    reg_eval_results = evaluate_regression_models(df)
    cls_eval_results = evaluate_classification_models(df)
    
    print("\nRegression models cross-validation results:")
    print(reg_eval_results[['R²', 'RMSE', 'MAE']])
    
    print("\nClassification models cross-validation results:")
    print(cls_eval_results[['Accuracy', 'F1 Score', 'Precision', 'Recall']])
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
