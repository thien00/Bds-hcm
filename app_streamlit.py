import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from data.data_preprocessing import load_and_preprocess_data, create_price_category
from models.combined_predictor import CombinedHousePricePredictor, visualize_prediction
from models.model_accuracy import evaluate_regression_models, evaluate_classification_models, plot_regression_accuracy, plot_classification_accuracy, plot_class_specific_accuracy

# Set page config
st.set_page_config(
    page_title="Dự đoán giá nhà",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_prepare_data():
    # Use the comprehensive preprocessing function
    return load_and_preprocess_data('data/df.csv')

def prepare_sample_data(df):
    # Return a few sample rows from the dataset
    return df.head(10)

def create_preprocessor():
    numeric_features = ['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor']
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

def train_model(df, model_type, model_name):
    try:
        X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District']]
        
        if model_type == 'regression':
            y = df['Price']
            # Define models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            }
        else:  # classification
            y = df['Price_Category'].astype(int)
            # Handle potential missing categories
            unique_classes = np.unique(y)
            if len(unique_classes) == 2:  # Binary classification case
                mapping = {unique_classes.min(): 0, unique_classes.max(): 1}
                y = y.map(mapping)
            # Define models
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBClassifier(objective='multi:softmax', random_state=42)
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get preprocessor
        preprocessor = create_preprocessor()
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', models[model_name])
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        if model_type == 'regression':
            metrics = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        else:
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Classification Report': classification_report(y_test, y_pred, output_dict=True)
            }
        
        return pipeline, X_test, y_test, y_pred, metrics
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

def plot_regression_results(X_test, y_test, y_pred):
    # Create DataFrame for plotting
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Plotly scatter plot
    fig = px.scatter(results_df, x='Actual', y='Predicted',
                    labels={'Actual': 'Actual Price (billion VND)', 'Predicted': 'Predicted Price (billion VND)'},
                    title='Actual vs Predicted Prices',
                    opacity=0.6)
    
    # Add diagonal line (perfect prediction)
    fig.add_trace(go.Scatter(
        x=[results_df['Actual'].min(), results_df['Actual'].max()],
        y=[results_df['Actual'].min(), results_df['Actual'].max()],
        mode='lines',
        name='Perfect prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        width=800,
        height=500,
    )
    
    st.plotly_chart(fig)
    
    # Distribution of errors
    errors = y_test - y_pred
    fig = px.histogram(errors, nbins=50, 
                     labels={'value': 'Error (billion VND)'},
                     title='Distribution of Prediction Errors')
    st.plotly_chart(fig)
    
    # Error vs Actual Price
    error_df = pd.DataFrame({'Actual': y_test, 'Error': errors})
    fig = px.scatter(error_df, x='Actual', y='Error',
                   labels={'Actual': 'Actual Price (billion VND)', 'Error': 'Error (billion VND)'},
                   title='Error vs Actual Price')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

def plot_classification_results(y_test, y_pred):
    # Compute union of classes present in the data
    import numpy as np
    classes = np.unique(np.concatenate([y_test, y_pred]))
    # Mapping for all possible classes
    mapping = {
        0: 'Low (<1B)',
        1: 'Medium (1-3B)',
        2: 'High (3-7B)',
        3: 'Very High (>7B)'
    }
    # Use only labels for the classes present
    class_labels = [mapping[c] for c in classes]
    
    # Compute confusion matrix using explicit labels
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Create heatmap with correct labels
    import plotly.express as px
    fig = px.imshow(cm, 
                    x=class_labels, 
                    y=class_labels,
                    labels=dict(x="Predicted Category", y="True Category", color="Count"),
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title='Confusion Matrix')
    st.plotly_chart(fig)
    
    # Classification accuracy by category
    accuracy_by_class = {}
    for i in range(len(class_labels)):
        mask = (y_test == i)
        if mask.sum() > 0:  # Check if we have samples in this class
            accuracy = np.mean(y_pred[mask] == y_test[mask])
            accuracy_by_class[class_labels[i]] = accuracy
    
    # Convert to DataFrame for plotting
    accuracy_df = pd.DataFrame(list(accuracy_by_class.items()), columns=['Category', 'Accuracy'])
    
    fig = px.bar(accuracy_df, x='Category', y='Accuracy',
               labels={'Accuracy': 'Classification Accuracy', 'Category': 'Price Category'},
               title='Classification Accuracy by Price Category')
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig)

def display_feature_importance(model, preprocessor):
    if hasattr(model, 'feature_importances_'):
        # Get feature names
        numeric_features = ['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor']
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
        
        # Plot feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(model.feature_importances_)],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)  # Top 15 features
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                   orientation='h',
                   title='Feature Importance',
                   labels={'Feature': 'Feature', 'Importance': 'Importance Score'})
        st.plotly_chart(fig)

def predict_with_model(pipeline, input_data):
    try:
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = pipeline.predict(input_df)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title("🏠 Dự đoán giá nhà")
    
    # Load data
    with st.spinner('Đang tải dữ liệu...'):
        df = load_and_prepare_data()
    
    # Create sidebar
    st.sidebar.header("Cài đặt")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Chọn trang",
        ["🔍 Khám phá dữ liệu", "📊 Huấn luyện & Đánh giá mô hình", "📈 So sánh độ chính xác", "🏡 Dự đoán giá"]
    )
    
    if page == "🔍 Khám phá dữ liệu":
        st.header("Khám phá dữ liệu")
        
        # Display sample data
        st.subheader("Dữ liệu mẫu")
        sample_data = prepare_sample_data(df)
        st.dataframe(sample_data)
        
        # Price distribution
        st.subheader("Phân bố giá")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Price', nbins=50,
                             title='Phân bố giá',
                             labels={'Price': 'Giá (tỷ VND)', 'count': 'Số lượng'})
            fig.update_layout(xaxis_range=[0, 20])  # Limit to 20B for better visualization
            st.plotly_chart(fig)
        
        with col2:
            # Convert Price_Category to string type to avoid issues with narwhals
            category_counts = df['Price_Category'].value_counts().reset_index()
            category_counts.columns = ['Price_Category', 'Count']
            
            # Create mapping for category labels
            category_labels = {
                0: 'Low (<1B)',
                1: 'Medium (1-3B)',
                2: 'High (3-7B)',
                3: 'Very High (>7B)'
            }
            
            # Create a new column with the category labels
            category_counts['Category_Label'] = category_counts['Price_Category'].map(category_labels)
            
            # Create pie chart with the prepared data
            fig = px.pie(
                category_counts, 
                values='Count',
                names='Category_Label', 
                title='Phân bố các mức giá'
            )
            st.plotly_chart(fig)
        
        # Price vs Acreage
        st.subheader("Giá so với Diện tích")
        fig = px.scatter(df.sample(1000) if len(df) > 1000 else df, 
                       x='Acreage', y='Price',
                       color='Price_Category',
                       color_continuous_scale=px.colors.sequential.Viridis,
                       labels={'Acreage': 'Diện tích (m²)', 'Price': 'Giá (tỷ VND)'},
                       title='Giá so với Diện tích')
        st.plotly_chart(fig)
        
        # Price by District
        st.subheader("Giá theo Quận")
        top_districts = df['District'].value_counts().nlargest(15).index
        district_data = df[df['District'].isin(top_districts)]
        
        fig = px.box(district_data, x='District', y='Price',
                   labels={'District': 'Quận', 'Price': 'Giá (tỷ VND)'},
                   title='Phân bố giá theo quận')
        fig.update_layout(xaxis={'categoryorder':'mean descending'})
        st.plotly_chart(fig)
        
        # Correlation matrix
        st.subheader("Tương quan các đặc trưng")
        corr_data = df[['Price', 'Acreage', 'Num_bedroom', 'Num_WC', 'Price_Category']].corr()
        fig = px.imshow(corr_data,
                      text_auto=True,
                      title='Ma trận tương quan',
                      labels=dict(color="Hệ số tương quan"),
                      color_continuous_scale='RdBu_r',
                      zmin=-1, zmax=1)
        st.plotly_chart(fig)
        
    elif page == "📊 Huấn luyện & Đánh giá mô hình":
        st.header("Huấn luyện & Đánh giá mô hình")
        
        # Select model type
        model_type = st.sidebar.radio("Chọn loại mô hình", ["Hồi quy", "Phân loại"])
        
        if model_type == "Hồi quy":
            st.subheader("Mô hình Hồi quy")
            # Select model
            model_name = st.sidebar.selectbox(
                "Chọn mô hình Hồi quy",
                ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
            )
            
            # Train button
            if st.sidebar.button("Huấn luyện mô hình"):
                with st.spinner(f'Đang huấn luyện {model_name}...'):
                    pipeline, X_test, y_test, y_pred, metrics = train_model(
                        df, 'regression', model_name
                    )
                
                # Display metrics
                st.subheader("Đánh giá mô hình")
                metrics_df = pd.DataFrame({
                    'Chỉ số': list(metrics.keys()),
                    'Giá trị': list(metrics.values())
                })
                st.table(metrics_df.set_index('Chỉ số'))
                
                # Plot results
                st.subheader("Kết quả dự đoán")
                plot_regression_results(X_test, y_test, y_pred)
                
                # Display feature importance if available
                st.subheader("Tầm quan trọng của đặc trưng")
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    display_feature_importance(
                        pipeline.named_steps['model'], 
                        pipeline.named_steps['preprocessor']
                    )
                else:
                    st.info("Tầm quan trọng của đặc trưng không khả dụng đối với mô hình này.")
        
        else:  # Classification
            st.subheader("Mô hình Phân loại")
            # Select model
            model_name = st.sidebar.selectbox(
                "Chọn mô hình Phân loại",
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
            )
            
            # Train button
            if st.sidebar.button("Huấn luyện mô hình"):
                with st.spinner(f'Đang huấn luyện {model_name}...'):
                    pipeline, X_test, y_test, y_pred, metrics = train_model(
                        df, 'classification', model_name
                    )
                
                # Display metrics
                st.subheader("Đánh giá mô hình")
                st.write(f"Độ chính xác: {metrics['Accuracy']:.4f}")
                
                # Classification report
                report = metrics['Classification Report']
                class_names = ['Thấp (<1B)', 'Trung bình (1-3B)', 'Cao (3-7B)', 'Rất cao (>7B)']
                
                # Fix: Handle missing categories in the report
                report_data = {
                    'Precision': [],
                    'Recall': [],
                    'F1-Score': [],
                    'Support': []
                }
                
                available_categories = []
                
                for i in range(4):
                    if str(i) in report:
                        report_data['Precision'].append(report[str(i)]['precision'])
                        report_data['Recall'].append(report[str(i)]['recall'])
                        report_data['F1-Score'].append(report[str(i)]['f1-score'])
                        report_data['Support'].append(report[str(i)]['support'])
                        available_categories.append(class_names[i])
                    else:
                        # If category is missing, add 0 or NaN values
                        report_data['Precision'].append(0)
                        report_data['Recall'].append(0)
                        report_data['F1-Score'].append(0)
                        report_data['Support'].append(0)
                        available_categories.append(class_names[i] + " (không có mẫu)")
                
                report_df = pd.DataFrame(report_data, index=available_categories)
                st.table(report_df)
                
                # Plot results
                st.subheader("Kết quả phân loại")
                plot_classification_results(y_test, y_pred)
                
                # Display feature importance if available
                st.subheader("Tầm quan trọng của đặc trưng")
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    display_feature_importance(
                        pipeline.named_steps['model'], 
                        pipeline.named_steps['preprocessor']
                    )
                else:
                    st.info("Tầm quan trọng của đặc trưng không khả dụng đối với mô hình này.")
    
    elif page == "📈 So sánh độ chính xác":
        st.header("So sánh độ chính xác của các thuật toán")
        
        with st.spinner("Đang tính toán độ chính xác..."):
            # Cache the evaluation results
            @st.cache_data(ttl=3600)  # Cache for 1 hour
            def get_model_accuracy():
                reg_results = evaluate_regression_models(df)
                cls_results = evaluate_classification_models(df)
                return reg_results, cls_results
            
            reg_results, cls_results = get_model_accuracy()
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Hồi quy", "Phân loại", "Chi tiết theo mức giá"])
            
            with tab1:
                st.subheader("Độ chính xác của các mô hình hồi quy")
                
                # Display metrics table - FIX: Changed Viridis to viridis (lowercase)
                st.dataframe(
                    reg_results[['R²', 'RMSE', 'MAE']].style.format({
                        'R²': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'MAE': '{:.4f}'
                    }).background_gradient(cmap='viridis', subset=['R²']).background_gradient(
                        cmap='viridis_r', subset=['RMSE', 'MAE']
                    ),
                    use_container_width=True
                )
                
                # Plot visualization
                st.plotly_chart(plot_regression_accuracy(reg_results))
                
                # Interpretation
                st.subheader("Phân tích")
                st.write("""
                - **R² (Hệ số xác định)**: Càng gần 1 càng tốt. Đo lường phần biến thiên trong dữ liệu được giải thích bởi mô hình.
                - **RMSE (Root Mean Squared Error)**: Càng thấp càng tốt. Đo lường sai số dự đoán trung bình (tính bằng tỷ đồng).
                - **MAE (Mean Absolute Error)**: Càng thấp càng tốt. Đo lường sai số tuyệt đối trung bình (tính bằng tỷ đồng).
                
                Dựa vào bảng kết quả:
                """)
                
                # Identify best model based on R²
                best_reg_model = reg_results['R²'].idxmax()
                st.write(f"- Mô hình **{best_reg_model}** có hiệu suất tốt nhất dựa trên chỉ số R² ({reg_results.loc[best_reg_model, 'R²']:.4f}).")
                
                # Identify best model based on RMSE
                best_rmse_model = reg_results['RMSE'].idxmin()
                st.write(f"- Mô hình **{best_rmse_model}** có sai số dự đoán thấp nhất với RMSE = {reg_results.loc[best_rmse_model, 'RMSE']:.4f} tỷ đồng.")
            
            with tab2:
                st.subheader("Độ chính xác của các mô hình phân loại")
                
                # Display metrics table
                st.dataframe(
                    cls_results[['Accuracy', 'F1 Score', 'Precision', 'Recall']].style.format({
                        'Accuracy': '{:.4f}',
                        'F1 Score': '{:.4f}',
                        'Precision': '{:.4f}',
                        'Recall': '{:.4f}'
                    }).background_gradient(cmap='viridis'),
                    use_container_width=True
                )
                
                # Plot visualization
                st.plotly_chart(plot_classification_accuracy(cls_results))
                
                # Interpretation
                st.subheader("Phân tích")
                st.write("""
                - **Accuracy (Độ chính xác)**: Tỷ lệ dự đoán đúng trên tổng số dự đoán.
                - **F1 Score**: Trung bình điều hòa của precision và recall, giải quyết vấn đề mất cân bằng dữ liệu.
                - **Precision (Độ chính xác)**: Trong các trường hợp được dự đoán là dương tính, bao nhiêu thực sự đúng.
                - **Recall (Độ nhạy)**: Trong số các trường hợp thực sự dương tính, bao nhiêu được nhận diện đúng.
                
                Dựa vào bảng kết quả:
                """)
                
                # Identify best model based on Accuracy
                best_cls_model = cls_results['Accuracy'].idxmax()
                st.write(f"- Mô hình **{best_cls_model}** có độ chính xác cao nhất ({cls_results.loc[best_cls_model, 'Accuracy']:.4f}).")
                
                # Identify best model based on F1
                best_f1_model = cls_results['F1 Score'].idxmax()
                st.write(f"- Mô hình **{best_f1_model}** có F1 Score cao nhất ({cls_results.loc[best_f1_model, 'F1 Score']:.4f}).")
            
            with tab3:
                st.subheader("Độ chính xác phân loại theo mức giá")
                
                # Compute and plot class-specific accuracy
                with st.spinner("Đang tính toán độ chính xác theo mức giá..."):
                    fig = plot_class_specific_accuracy(df)
                    st.plotly_chart(fig)
                
                st.write("""
                Biểu đồ trên cho thấy độ chính xác của từng mô hình đối với các mức giá khác nhau.
                Một số mô hình có thể hoạt động tốt với mức giá thấp nhưng lại kém chính xác với mức giá cao, hoặc ngược lại.
                """)
                
                st.info("""
                **Ghi chú**: Độ chính xác có thể thấp ở một số nhóm do:
                - Số lượng mẫu không đồng đều giữa các mức giá
                - Đặc trưng của nhà ở mức giá cao thường đa dạng hơn và khó dự đoán hơn
                - Một số nhóm có thể có quá ít dữ liệu để mô hình học hiệu quả
                """)
    
    else:  # Price Prediction
        st.header("Dự đoán giá nhà")
        
        # Use combined predictor or separate models
        prediction_approach = st.sidebar.radio(
            "Phương pháp dự đoán",
            ["Tích hợp (Hồi quy + Phân loại)", "Chỉ Hồi quy", "Chỉ Phân loại"]
        )
        
        # Add model selection based on approach - MOVED OUTSIDE THE BUTTON CLICK HANDLER
        selected_model = None
        if prediction_approach == "Chỉ Hồi quy":
            # Import regression model components
            from features.feature_engineering import create_preprocessor
            from models.regression_models import get_regression_models
            
            # Let user select regression model
            reg_models = get_regression_models()
            selected_model = st.sidebar.selectbox(
                "Chọn mô hình hồi quy",
                list(reg_models.keys()),
                index=1  # Default to Random Forest
            )
        
        elif prediction_approach == "Chỉ Phân loại":
            # Import classification model components
            from features.feature_engineering import create_preprocessor
            from models.classification_models import get_classification_models
            
            # Let user select classification model
            cls_models = get_classification_models()
            selected_model = st.sidebar.selectbox(
                "Chọn mô hình phân loại",
                list(cls_models.keys()),
                index=1  # Default to Random Forest
            )
        
        # Input form for house features
        st.subheader("Nhập thông tin nhà")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0)
            num_bedrooms = st.number_input("Số phòng ngủ", min_value=0, max_value=20, value=2)
            num_floors = st.number_input("Số tầng", min_value=1, max_value=50, value=1)

        with col2:
            num_bathrooms = st.number_input("Số phòng tắm", min_value=0, max_value=20, value=2)
            district_list = sorted(df['District'].unique().tolist())
            district = st.selectbox("Quận", district_list, index=district_list.index('Quận 7') if 'Quận 7' in district_list else 0)
        
        # Create input data
        input_data = {
            'Acreage': area,
            'Num_bedroom': num_bedrooms,
            'Num_WC': num_bathrooms,
            'Num_floor': num_floors,
            'District': district
        }
        
        # Predict button
        if st.button("Dự đoán giá"):
            if prediction_approach == "Tích hợp (Hồi quy + Phân loại)":
                with st.spinner('Đang huấn luyện mô hình tích hợp...'):
                    predictor = CombinedHousePricePredictor()
                    predictor.train(df)
                    
                    prediction = predictor.predict(input_data)
                    
                    # Display prediction
                    st.subheader("Kết quả dự đoán giá")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Giá dự đoán", 
                            f"{prediction['predicted_price']:.2f} tỷ VND",
                            delta=None
                        )
                        st.write(f"**Khoảng tin cậy 90%:** {prediction['price_confidence_interval'][0]:.2f} - {prediction['price_confidence_interval'][1]:.2f} tỷ VND")
                    
                    with col2:
                        st.metric(
                            "Mức giá dự đoán",
                            prediction['predicted_category_name']
                        )
                        
                        # Display category probabilities
                        st.write("**Xác suất của từng mức giá:**")
                        for category, prob in prediction['category_probabilities'].items():
                            st.write(f"- {category}: {prob:.2f}")
                    
                    # Visualization
                    st.subheader("Biểu đồ dự đoán")
                    
                    # Create plots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot 1: Price prediction with confidence interval
                    price_pred = prediction['predicted_price']
                    lower_bound, upper_bound = prediction['price_confidence_interval']
                    
                    ax1.barh(['Giá'], [price_pred], color='skyblue', alpha=0.7, height=0.5)
                    ax1.errorbar([price_pred], [0], xerr=[[price_pred-lower_bound], [upper_bound-price_pred]], 
                                fmt='o', color='black', capsize=10)
                    ax1.set_xlim(0, upper_bound * 1.1)
                    ax1.set_title('Giá dự đoán với khoảng tin cậy 90%')
                    ax1.set_xlabel('Giá (tỷ VND)')
                    ax1.set_yticks([])
                    
                    # Price labels
                    ax1.text(price_pred, 0, f'{price_pred:.2f}B', 
                            horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
                    ax1.text(lower_bound, 0, f'{lower_bound:.2f}B', 
                            horizontalalignment='center', verticalalignment='top')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Similar properties
                    st.subheader("Các bất động sản tương tự")
                    
                    # Filter properties in the same price range and district
                    similar_props = df[
                        (df['District'] == district) & 
                        (df['Price'] >= prediction['price_confidence_interval'][0]) &
                        (df['Price'] <= prediction['price_confidence_interval'][1])
                    ].head(5)
                    
                    if len(similar_props) > 0:
                        st.dataframe(similar_props[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District', 'Price']])
                    else:
                        st.write("Không tìm thấy bất động sản tương tự.")
                    
            elif prediction_approach == "Chỉ Hồi quy":
                with st.spinner('Đang huấn luyện mô hình hồi quy...'):
                    # Create preprocessor
                    preprocessor = create_preprocessor()
                    
                    # Use the already selected model (we don't need to show the selection again)
                    # Create and train pipeline
                    from sklearn.pipeline import Pipeline
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', reg_models[selected_model])
                    ])
                    
                    # Convert input_data to DataFrame for prediction
                    input_df = pd.DataFrame([input_data])
                    
                    # Fit model on data
                    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District']]
                    y = df['Price']
                    pipeline.fit(X, y)
                    
                    # Make prediction
                    predicted_price = pipeline.predict(input_df)[0]
                    
                    # Display results
                    st.subheader("Kết quả dự đoán giá")
                    
                    st.metric(
                        "Giá dự đoán", 
                        f"{predicted_price:.2f} tỷ VND",
                        delta=None
                    )
                    
                    # Simple confidence interval (±15%)
                    lower_bound = predicted_price * 0.85
                    upper_bound = predicted_price * 1.15
                    
                    st.write(f"**Khoảng tin cậy đơn giản:** {lower_bound:.2f} - {upper_bound:.2f} tỷ VND")
                    
                    # Create a simple visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(['Giá'], [predicted_price], color='skyblue', alpha=0.7, height=0.5)
                    ax.errorbar([predicted_price], [0], xerr=[[predicted_price-lower_bound], [upper_bound-predicted_price]], 
                               fmt='o', color='black', capsize=10)
                    ax.set_xlim(0, upper_bound * 1.1)
                    ax.set_title(f'Giá dự đoán - Mô hình {selected_model}')
                    ax.set_xlabel('Giá (tỷ VND)')
                    ax.set_yticks([])
                    ax.text(predicted_price, 0, f'{predicted_price:.2f}B', 
                           horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add similar properties section
                    st.subheader("Các bất động sản tương tự")
                    
                    # Filter properties in the same price range and district
                    similar_props = df[
                        (df['District'] == district) & 
                        (df['Price'] >= lower_bound) &
                        (df['Price'] <= upper_bound)
                    ].head(5)
                    
                    if len(similar_props) > 0:
                        st.dataframe(similar_props[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District', 'Price']])
                        
                        # Add a visualization comparing the prediction with similar properties
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Plot prices of similar properties
                        ax.bar(range(len(similar_props)), similar_props['Price'], color='lightgray', 
                               alpha=0.7, label='Nhà tương tự')
                        
                        # Plot prediction as a horizontal line
                        ax.axhline(y=predicted_price, color='red', linestyle='-', 
                                  label=f'Dự đoán: {predicted_price:.2f}B')
                        
                        # Add price labels above each bar
                        for i, price in enumerate(similar_props['Price']):
                            ax.text(i, price + 0.1, f'{price:.2f}B', ha='center')
                        
                        ax.set_ylabel('Giá (tỷ VND)')
                        ax.set_title('So sánh với các bất động sản tương tự')
                        ax.set_xticks(range(len(similar_props)))
                        ax.set_xticklabels([f'{a}m², {b}PN, {c}WC' for a, b, c in 
                                          zip(similar_props['Acreage'], similar_props['Num_bedroom'], similar_props['Num_WC'])],
                                         rotation=45, ha='right')
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Không tìm thấy bất động sản tương tự.")
            
            elif prediction_approach == "Chỉ Phân loại":
                with st.spinner('Đang huấn luyện mô hình phân loại...'):
                    # Create preprocessor
                    preprocessor = create_preprocessor()
                    
                    # Use the already selected model (we don't need to show the selection again)
                    # Create and train pipeline
                    from sklearn.pipeline import Pipeline
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', cls_models[selected_model])
                    ])
                    
                    # Convert input_data to DataFrame for prediction
                    input_df = pd.DataFrame([input_data])
                    
                    # Define price categories for training
                    conditions = [
                        (df['Price'] < 1),
                        (df['Price'] >= 1) & (df['Price'] < 3),
                        (df['Price'] >= 3) & (df['Price'] < 7),
                        (df['Price'] >= 7)
                    ]
                    choices = [0, 1, 2, 3]
                    df['Price_Category'] = np.select(conditions, choices, default=np.nan)
                    
                    # Fit model on data
                    X = df[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District']]
                    y = df['Price_Category'].astype(int)
                    pipeline.fit(X, y)
                    
                    # Make prediction
                    predicted_class = pipeline.predict(input_df)[0]
                    
                    # Get probabilities
                    price_categories = {
                        0: 'Low price (< 1B)',
                        1: 'Medium price (1-3B)',
                        2: 'High price (3-7B)',
                        3: 'Very high price (> 7B)'
                    }
                    
                    try:
                        proba = pipeline.predict_proba(input_df)[0]
                        category_probs = {price_categories[i]: prob for i, prob in enumerate(proba)}
                    except:
                        category_probs = {price_categories[predicted_class]: 1.0}
                    
                    # Display results
                    st.subheader("Kết quả dự đoán mức giá")
                    
                    st.metric(
                        "Mức giá dự đoán", 
                        price_categories[predicted_class],
                        delta=None
                    )
                    
                    # Display category probabilities
                    st.write("**Xác suất của từng mức giá:**")
                    for category, prob in category_probs.items():
                        st.write(f"- {category}: {prob:.2f}")
                    
                    # Create a visualization of the probability distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    categories = list(category_probs.keys())
                    probabilities = list(category_probs.values())
                    
                    bars = ax.bar(categories, probabilities, color=['lightgreen', 'skyblue', 'orange', 'salmon'])
                    
                    # Highlight the predicted class
                    bars[predicted_class].set_color('darkblue')
                    
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel('Xác suất')
                    ax.set_title(f'Phân bố xác suất các mức giá - Mô hình {selected_model}')
                    
                    # Add probability labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add similar properties section
                    st.subheader("Các bất động sản tương tự")
                    
                    # Map predicted class to price range
                    price_ranges = {
                        0: (0, 1),       # Low: < 1B
                        1: (1, 3),       # Medium: 1-3B
                        2: (3, 7),       # High: 3-7B
                        3: (7, float('inf'))  # Very High: > 7B
                    }
                    
                    min_price, max_price = price_ranges[predicted_class]
                    
                    # Filter properties in the same category and district
                    similar_props = df[
                        (df['District'] == district) & 
                        (df['Price'] >= min_price) &
                        (df['Price'] <= max_price)
                    ].head(5)
                    
                    if len(similar_props) > 0:
                        st.dataframe(similar_props[['Acreage', 'Num_bedroom', 'Num_WC', 'Num_floor', 'District', 'Price']])
                        
                        # Add a visualization of similar properties in this category
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Plot scatter of similar properties by price vs size
                        scatter = ax.scatter(similar_props['Acreage'], similar_props['Price'], 
                                           s=similar_props['Num_bedroom']*30, alpha=0.7,
                                           c=similar_props['Num_WC'], cmap='viridis')
                        
                        # Highlight the user's input
                        ax.scatter([area], [(min_price + max_price)/2], marker='*', color='red', s=200,
                                 label='Dự đoán của bạn')
                        
                        # Add a legend for bedroom size
                        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, 
                                                                num=sorted(similar_props['Num_bedroom'].unique()))
                        bedroom_legend = ax.legend(handles, 
                                                 [f'{int(size/30)} phòng ngủ' for size in sorted(similar_props['Num_bedroom'].unique()*30)],
                                                 loc="upper left", title="Số phòng ngủ")
                        ax.add_artist(bedroom_legend)
                        
                        # Add a colorbar for bathroom count
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Số phòng tắm')
                        
                        ax.set_title(f'Các bất động sản trong mức giá: {price_categories[predicted_class]}')
                        ax.set_xlabel('Diện tích (m²)')
                        ax.set_ylabel('Giá (tỷ VND)')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Không tìm thấy bất động sản tương tự.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Dự án Dự đoán Giá Nhà
        
        Kết hợp phương pháp hồi quy và phân loại để dự đoán giá nhà.
        """
    )

if __name__ == "__main__":
    main()
