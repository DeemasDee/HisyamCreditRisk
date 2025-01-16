import streamlit as st
import joblib
import numpy as np

# Memuat model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Pastikan 'rf_best' disimpan sebagai model.pkl

model = load_model()

def main():
    st.title("Loan Default Prediction Dashboard")

    # Input fields
    st.subheader("Input Details")
    
    # Loan Grade Selection
    loan_grade = st.selectbox(
        "Loan Grade",
        options=["A", "B", "C", "D"],
        index=0
    )

    # Loan Intent Input
    loan_intent = st.text_input(
        "Loan Intent",
        placeholder="Enter loan intent (e.g., STUDENT, rent)"
    )

    # Location Input
    location = st.text_input(
        "Location",
        placeholder="Enter location (e.g., Jakarta)"
    )

    # Button to predict
    if st.button("Predict"):
        # Format input data sesuai dengan model
        # Konversi input ke numerik jika diperlukan berdasarkan preprocessing
        grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4}  # Mapping contoh
        loan_grade_num = grade_mapping.get(loan_grade, 0)
        
        # Preprocessing tambahan (contoh, sesuaikan dengan model Anda)
        # Jika ada encoder untuk loan_intent atau location, tambahkan di sini
        
        # Buat array input
        input_data = np.array([[loan_grade_num, loan_intent, location]])  # Sesuaikan kolom

        # Lakukan prediksi
        try:
            prediction = model.predict(input_data)
            prediction_result = "Yes" if prediction[0] == 1 else "No"
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

        # Display Prediction
        st.subheader("Prediction")
        st.write(f"Default: {prediction_result}")

if __name__ == "__main__":
    main()
