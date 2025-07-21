import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

def plot_regression_results(y_test, y_pred):
    """Create plots for regression model evaluation"""
    # Create DataFrame for plotting
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Scatter plot for actual vs predicted
    fig_scatter = px.scatter(results_df, x='Actual', y='Predicted',
                        labels={'Actual': 'Actual Price (billion VND)', 
                                'Predicted': 'Predicted Price (billion VND)'},
                        title='Actual vs Predicted Prices',
                        opacity=0.6)
    
    # Add diagonal line (perfect prediction)
    fig_scatter.add_trace(go.Scatter(
        x=[results_df['Actual'].min(), results_df['Actual'].max()],
        y=[results_df['Actual'].min(), results_df['Actual'].max()],
        mode='lines',
        name='Perfect prediction',
        line=dict(color='red', dash='dash')
    ))
    
    # Distribution of errors
    errors = y_test - y_pred
    fig_hist = px.histogram(errors, nbins=50, 
                        labels={'value': 'Error (billion VND)'},
                        title='Distribution of Prediction Errors')
    
    # Error vs Actual Price
    error_df = pd.DataFrame({'Actual': y_test, 'Error': errors})
    fig_error = px.scatter(error_df, x='Actual', y='Error',
                      labels={'Actual': 'Actual Price (billion VND)', 
                              'Error': 'Error (billion VND)'},
                      title='Error vs Actual Price')
    fig_error.add_hline(y=0, line_dash="dash", line_color="red")
    
    return fig_scatter, fig_hist, fig_error

def plot_confusion_matrix(y_test, y_pred, class_names=None):
    """Create confusion matrix plot for classification model evaluation"""
    # Get unique classes
    classes = np.unique(np.concatenate([y_test, y_pred]))
    
    # Define default class names if not provided
    if class_names is None:
        class_names = {
            0: 'Low (<1B)',
            1: 'Medium (1-3B)',
            2: 'High (3-7B)',
            3: 'Very High (>7B)'
        }
    
    # Get labels for the classes present
    labels = [class_names[c] for c in classes]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Create heatmap
    fig = px.imshow(cm, 
                x=labels, 
                y=labels,
                labels=dict(x="Predicted Category", y="True Category", color="Count"),
                text_auto=True,
                color_continuous_scale='Blues',
                title='Confusion Matrix')
    
    return fig

def plot_classification_accuracy_by_class(y_test, y_pred, class_names=None):
    """Plot classification accuracy for each class"""
    if class_names is None:
        class_names = {
            0: 'Low (<1B)',
            1: 'Medium (1-3B)',
            2: 'High (3-7B)',
            3: 'Very High (>7B)'
        }
    
    # Get unique classes in the test data
    classes = np.unique(y_test)
    
    # Calculate accuracy by class
    accuracy_by_class = {}
    for cls in classes:
        mask = (y_test == cls)
        if mask.sum() > 0:  # Make sure we have samples in this class
            accuracy = np.mean(y_pred[mask] == y_test[mask])
            accuracy_by_class[class_names[cls]] = accuracy
    
    # Convert to DataFrame for plotting
    accuracy_df = pd.DataFrame(list(accuracy_by_class.items()), 
                              columns=['Category', 'Accuracy'])
    
    fig = px.bar(accuracy_df, x='Category', y='Accuracy',
              labels={'Accuracy': 'Classification Accuracy', 
                      'Category': 'Price Category'},
              title='Classification Accuracy by Price Category')
    fig.update_layout(yaxis_range=[0, 1])
    
    return fig

def plot_regression_accuracy(reg_results):
    """Plot regression model comparison"""
    # Create a bar chart for R² scores
    fig = px.bar(
        reg_results.reset_index(), 
        x='index', 
        y='R²', 
        error_y='R² Std',
        title='Regression Models Performance (R²)',
        labels={'index': 'Model', 'R²': 'R² Score'},
        color='R²',
        color_continuous_scale='viridis'
    )
    
    # Add a horizontal line for R²=0.7 as "good model" threshold
    fig.add_shape(
        type="line",
        x0=-0.5, y0=0.7, x1=3.5, y1=0.7,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def plot_classification_accuracy(cls_results):
    """Plot classification model comparison"""
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
    fig.add_shape(
        type="line",
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
    # Prepare data
    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'District']]
    y = df['Price_Category'].astype(int)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Import here to avoid circular imports
    from features.feature_engineering import create_preprocessor
    from models.classification_models import get_classification_models
    
    # Get preprocessor
    preprocessor = create_preprocessor()
    
    # Get classification models
    models = get_classification_models()
    
    # Category labels mapping
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
        
        from sklearn.pipeline import Pipeline
        
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
                class_acc = np.mean(y_pred[idx] == y_test[idx])
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
        color_continuous_scale="viridis"
    )
    
    return fig
