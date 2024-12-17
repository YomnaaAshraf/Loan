import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from scipy.stats import mstats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree

# App Title
st.title("Loan Prediction Analysis and Modeling")

# Sidebar for uploading data
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Dataset Information:")
    st.write(df.info())

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning")
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Feature Engineering
    st.subheader("Feature Engineering")
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome'] = np.log(df['TotalIncome'])
    df['LoanAmount'] = np.log(df['LoanAmount'])
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    st.write("Distribution of Total Income")
    fig, ax = plt.subplots()
    df['TotalIncome'].hist(bins=20, ax=ax)
    st.pyplot(fig)

    st.write("Distribution of Loan Amount")
    fig, ax = plt.subplots()
    df['LoanAmount'].hist(bins=20, ax=ax)
    st.pyplot(fig)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    st.write("Total Income vs Loan Amount")
    fig, ax = plt.subplots()
    plt.scatter(df['TotalIncome'], df['LoanAmount'])
    plt.xlabel('Total Income')
    plt.ylabel('Loan Amount')
    plt.title('Total Income vs Loan Amount')
    st.pyplot(fig)

    st.write("Credit History vs Total Income")
    fig, ax = plt.subplots()
    plt.scatter(df['Credit_History'], df['TotalIncome'])
    plt.xlabel('Credit History')
    plt.ylabel('Total Income')
    plt.title('Credit History vs Total Income')
    st.pyplot(fig)

    st.write("Average Loan Amount by Property Area")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Property_Area', y='LoanAmount', ci=None, ax=ax)
    plt.xlabel('Property Area')
    plt.ylabel('Average Loan Amount')
    plt.title('Average Loan Amount by Property Area')
    st.pyplot(fig)

    # Heatmap for Education vs Loan Status
    st.write("Education vs Loan Status")
    cross_tab = pd.crosstab(df['Education'], df['Loan_Status'])
    fig, ax = plt.subplots()
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Heatmap for Self Employed vs Loan Status
    st.write("Self Employed vs Loan Status")
    cross_tab = pd.crosstab(df['Self_Employed'], df['Loan_Status'])
    fig, ax = plt.subplots()
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numerical_columns = ['TotalIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    correlation_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

    # Model Selection and Training
    st.subheader("Modeling")
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents'])
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    def create_nn_model():
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# Custom Keras Classifier for KFold
    class KerasNNClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, epochs=10, batch_size=32):
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def fit(self, X, y):
            self.model = create_nn_model()
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict(self, X):
            return (self.model.predict(X) > 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold CV
        
    model_choice = st.selectbox("Select a Model", ["Decision Tree", "Naive Bayes", "Gradient Boosting"])
    
    if model_choice == "Decision Tree":
        model = Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=42))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Decision Tree Model Accuracy: {accuracy:.2f}")
        st.write("Decision Tree Visualization:")
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(model.named_steps['classifier'], feature_names=X_train.columns, filled=True, ax=ax)
        st.pyplot(fig)
    
    elif model_choice == "Naive Bayes":
        model = Pipeline([('scaler', StandardScaler()), ('classifier', GaussianNB())])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Naive Bayes Model Accuracy: {accuracy:.2f}")
    
   
    elif model_choice == "Gradient Boosting":
    # Define K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create a pipeline with StandardScaler and GradientBoostingClassifier
        pipeline_gb = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: Apply StandardScaler
            ('model', GradientBoostingClassifier(random_state=42))  # Step 2: Apply GradientBoostingClassifier
        ])
        
        # 1. Perform cross-validation and evaluate with K-Fold
        cv_scores_gb = cross_val_score(pipeline_gb, X_train, y_train, cv=kf)  # K-Fold cross-validation
        st.write("K-Fold Cross-Validation Results:")
        st.write(f"Cross-validation scores for each fold: {cv_scores_gb}")
        st.write(f"Mean CV accuracy (K-Fold): {np.mean(cv_scores_gb):.4f}")
        
        # 2. Fit the pipeline on the training data
        pipeline_gb.fit(X_train, y_train)
        
        # Predict on test data
        y_pred_gb_test = pipeline_gb.predict(X_test)
        accuracy_gb_test = accuracy_score(y_test, y_pred_gb_test)
        st.write(f"Accuracy on Test Set: {accuracy_gb_test:.4f}")
        
        # 3. Cross-validated predictions on the entire dataset
        y_pred_gb_cv = cross_val_predict(pipeline_gb, X, y, cv=kf)
        overall_accuracy = accuracy_score(y, y_pred_gb_cv)
        st.write(f"Overall Accuracy (Cross-Validated Predictions): {overall_accuracy:.4f}")
        st.write("\nClassification Report:")
        st.text(classification_report(y, y_pred_gb_cv))
        
        # 4. Access and visualize the first tree in the GradientBoostingClassifier ensemble
        gb_model = pipeline_gb.named_steps['model']  # Extract the GradientBoostingClassifier model
        tree = gb_model.estimators_[0, 0]  # Access the first tree in the ensemble
        
        # Display the first tree
        st.write("First Tree in Gradient Boosting Ensemble:")
        st.write(tree)
        
        # Check if the first tree is a DecisionTreeRegressor
        if isinstance(tree, DecisionTreeRegressor):
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                tree, 
                filled=True, 
                feature_names=X_train.columns, 
                class_names=[str(cls) for cls in np.unique(y_train)], 
                fontsize=10
            )
            plt.title("First Tree in Gradient Boosting Classifier")
            st.pyplot(fig)
        else:
            st.write("The first tree is not a DecisionTreeRegressor.")
    