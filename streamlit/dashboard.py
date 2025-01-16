import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Load the saved model (replace 'model.pkl' with the actual model file name)
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file is in the correct location.")
        return None

# Function to load data
def load_data(file):
    try:
        df = pd.read_csv(file)
        st.write(f"Data Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Split input and output
def split_input_output(data, target_column):
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        st.write(f"Features Shape: {X.shape}")
        st.write(f"Target Shape: {y.shape}")
        return X, y
    except KeyError:
        st.error(f"Column '{target_column}' not found in data.")
        return None, None

# Main Streamlit app
def main():
    st.title("Prediction Dashboard")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        if data is not None:
            st.write("### Preview of Uploaded Data")
            st.dataframe(data.head())

            # Select target column
            target_column = st.selectbox("Select the target column", data.columns)

            if target_column:
                X, y = split_input_output(data, target_column)

                # Load model
                model = load_model()

                if model is not None and X is not None:
                    # Make predictions
                    if st.button("Make Predictions"):
                        predictions = model.predict(X)
                        st.write("### Predictions")
                        st.write(predictions)

if __name__ == "__main__":
    main()
