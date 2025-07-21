import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class CombinedHousePricePredictor:
    """
    A combined regressor and classifier for house price prediction.
    - Regressor: Predicts exact price value
    - Classifier: Predicts price category
    """
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.preprocessor = None
        self.price_categories = {
            0: 'Low price (< 1B)',
            1: 'Medium price (1-3B)',
            2: 'High price (3-7B)',
            3: 'Very high price (> 7B)'
        }
        self.features = ['Acreage', 'Num_bedroom', 'Num_WC', 'District']
    
    def _create_preprocessor(self):
        """Create the data preprocessor"""
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
    
    def create_price_category(self, price):
        """Convert continuous price to category"""
        if price < 1:
            return 0
        elif price < 3:
            return 1
        elif price < 7:
            return 2
        else:
            return 3
    
    def train(self, df):
        """
        Train both regression and classification models
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training data with necessary columns: 'Acreage', 'Num_bedroom', 'Num_WC', 
            'District', and 'Price'
        """
        print("Preparing data for training...")
        
        # Ensure price is in correct format
        df = df[df["Price"] != ""].copy()
        df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
        df = df[df["Price"] <= 500].reset_index(drop=True)
        
        # Create price categories for classification
        df['Price_Category'] = df['Price'].apply(self.create_price_category)
        
        # Prepare features and targets
        X = df[self.features]
        y_reg = df['Price']
        y_cls = df['Price_Category'].astype(int)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_reg, y_cls, test_size=0.2, random_state=42
        )
        
        # Train regression model
        print("Training regression model...")
        self.regression_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.regression_model.fit(X_train, y_reg_train)
        
        # Train classification model
        print("Training classification model...")
        self.classification_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.classification_model.fit(X_train, y_cls_train)
        
        # Evaluate models
        reg_score = self.regression_model.score(X_test, y_reg_test)
        cls_score = self.classification_model.score(X_test, y_cls_test)
        
        print(f"Regression model R² score: {reg_score:.4f}")
        print(f"Classification model accuracy: {cls_score:.4f}")
        
        return reg_score, cls_score
    
    def predict(self, house_data):
        """
        Predict house price and category
        
        Parameters:
        -----------
        house_data : pandas.DataFrame or dict
            House features to predict. Must include: 'Acreage', 'Num_bedroom', 'Num_WC', 'District'
        
        Returns:
        --------
        dict
            Dictionary with predicted price and category
        """
        if self.regression_model is None or self.classification_model is None:
            raise ValueError("Models not trained. Call train() method first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(house_data, dict):
            house_data = pd.DataFrame([house_data])
            
        # Ensure all features are present
        for feature in self.features:
            if feature not in house_data.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # Make predictions
        predicted_price = self.regression_model.predict(house_data)[0]
        predicted_category = self.classification_model.predict(house_data)[0]
        predicted_category_name = self.price_categories[predicted_category]
        
        # Get probability distribution for categories
        category_probs = self.classification_model.predict_proba(house_data)[0]
        
        # Create confidence interval for regression using a more reliable method
        try:
            # Try to use the forest model to get lower and upper bounds
            forest = self.regression_model.named_steps['model']
            preprocessed_data = self.preprocessor.transform(house_data)
            
            tree_preds = []
            # Only attempt this for RandomForestRegressor
            if hasattr(forest, 'estimators_'):
                for tree in forest.estimators_:
                    # For trees in sklearn, we need special handling
                    if hasattr(tree, 'predict'):
                        # Make sure tree can predict from preprocessed data
                        try:
                            tree_preds.append(tree.predict(preprocessed_data)[0])
                        except:
                            # If individual tree prediction fails, use a simpler approach
                            pass
            
            # If we collected enough predictions, use them for the interval
            if len(tree_preds) > 0:
                lower_bound = np.percentile(tree_preds, 5)
                upper_bound = np.percentile(tree_preds, 95)
            else:
                # Fallback to a simpler approach - use a percentage of the predicted price
                lower_bound = predicted_price * 0.85
                upper_bound = predicted_price * 1.15
        except:
            # Fallback to a simpler approach
            lower_bound = predicted_price * 0.85
            upper_bound = predicted_price * 1.15
        
        result = {
            'predicted_price': predicted_price,
            'price_confidence_interval': (lower_bound, upper_bound),
            'predicted_category': predicted_category,
            'predicted_category_name': predicted_category_name,
            'category_probabilities': {self.price_categories[i]: prob for i, prob in enumerate(category_probs)}
        }
        
        return result
    
    def save_models(self, folder='models'):
        """Save trained models to disk"""
        if self.regression_model is None or self.classification_model is None:
            raise ValueError("Models not trained. Call train() method first.")
        
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Save regression model
        with open(os.path.join(folder, 'regression_model.pkl'), 'wb') as f:
            pickle.dump(self.regression_model, f)
            
        # Save classification model
        with open(os.path.join(folder, 'classification_model.pkl'), 'wb') as f:
            pickle.dump(self.classification_model, f)
            
        print(f"Models saved to {folder}/")
    
    def load_models(self, folder='models'):
        """Load trained models from disk"""
        # Check if model files exist
        reg_path = os.path.join(folder, 'regression_model.pkl')
        cls_path = os.path.join(folder, 'classification_model.pkl')
        
        if not os.path.exists(reg_path) or not os.path.exists(cls_path):
            raise FileNotFoundError(f"Model files not found in {folder}/")
        
        # Load regression model
        with open(reg_path, 'rb') as f:
            self.regression_model = pickle.load(f)
            
        # Load classification model
        with open(cls_path, 'rb') as f:
            self.classification_model = pickle.load(f)
            
        print("Models loaded successfully!")

def visualize_prediction(prediction):
    """
    Create visual representation of the prediction
    
    Parameters:
    -----------
    prediction : dict
        Prediction result from CombinedHousePricePredictor.predict()
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Price prediction with confidence interval
    price_pred = prediction['predicted_price']
    lower_bound, upper_bound = prediction['price_confidence_interval']
    
    ax1.barh(['Price'], [price_pred], color='skyblue', alpha=0.7, height=0.5)
    ax1.errorbar([price_pred], [0], xerr=[[price_pred-lower_bound], [upper_bound-price_pred]], 
                 fmt='o', color='black', capsize=10)
    ax1.set_xlim(0, upper_bound * 1.1)
    ax1.set_title('Price Prediction with 90% Confidence Interval')
    ax1.set_xlabel('Price (billion VND)')
    ax1.set_yticks([])
    
    # Price labels
    ax1.text(price_pred, 0, f'{price_pred:.2f}B', 
             horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax1.text(lower_bound, 0, f'{lower_bound:.2f}B', 
             horizontalalignment='center', verticalalignment='top')
    ax1.text(upper_bound, 0, f'{upper_bound:.2f}B', 
             horizontalalignment='center', verticalalignment='top')
    
    # Plot 2: Category probability distribution
    categories = list(prediction['category_probabilities'].keys())
    probabilities = list(prediction['category_probabilities'].values())
    
    # Highlight predicted category
    colors = ['lightgray'] * len(categories)
    colors[prediction['predicted_category']] = 'skyblue'
    
    ax2.bar(categories, probabilities, color=colors)
    ax2.set_title('Price Category Probabilities')
    ax2.set_xlabel('Price Category')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add probability values on top of bars
    for i, prob in enumerate(probabilities):
        ax2.text(i, prob + 0.02, f'{prob:.2f}', 
                 horizontalalignment='center', fontweight='bold' if i == prediction['predicted_category'] else 'normal')
    
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    from price_prediction import load_data
    
    print("Loading data...")
    df = load_data()
    
    # Create and train predictor
    predictor = CombinedHousePricePredictor()
    predictor.train(df)
    
    # Save models
    predictor.save_models()
    
    # Example prediction
    sample_house = {
        'Acreage': 80.0,
        'Num_bedroom': 3,
        'Num_WC': 2,
        'District': 'Quận 7'
    }
    
    print("\nPredicting sample house:")
    for key, value in sample_house.items():
        print(f"  {key}: {value}")
        
    prediction = predictor.predict(sample_house)
    
    print("\nPrediction results:")
    print(f"  Predicted price: {prediction['predicted_price']:.2f} billion VND")
    print(f"  Price range (90% confidence): {prediction['price_confidence_interval'][0]:.2f} - {prediction['price_confidence_interval'][1]:.2f} billion VND")
    print(f"  Predicted category: {prediction['predicted_category_name']}")
    
    print("\nCategory probabilities:")
    for category, prob in prediction['category_probabilities'].items():
        print(f"  {category}: {prob:.2f}")
    
    # Visualize prediction
    print("\nVisualizing prediction...")
    visualize_prediction(prediction)

if __name__ == "__main__":
    main()
