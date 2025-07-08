import streamlit as st
import numpy as np
import pickle

# Define the model wrapper class used in pickling
class CombinedLoanModel:
    def __init__(self, log_model, rf_model, encoders):
        self.log_model = log_model
        self.rf_model = rf_model
        self.encoders = encoders

    def predict(self, raw_input):  # expects list like [age, income, education, score, employment]
        edu = self.encoders["EducationLevel"].transform([raw_input[2]])[0]
        emp = self.encoders["EmploymentStatus"].transform([raw_input[4]])[0]
        input_data = np.array([[raw_input[0], raw_input[1], edu, raw_input[3], emp]])
        log_p = self.log_model.predict_proba(input_data)[0][1]
        rf_p = self.rf_model.predict_proba(input_data)[0][1]
        final_prob = (log_p + rf_p) / 2
        final_pred = int(final_prob >= 0.5)
        return final_pred, final_prob

# Load the combined model
@st.cache_resource
def load_model():
    with open("combined_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")
st.title("🏦 Loan Eligibility Predictor")
st.markdown("Use our smart ML model (Logistic + Random Forest) to check if a loan will be approved.")

st.markdown("---")

# Input Form
with st.form("loan_form"):
    age = st.slider("👤 Age", 18, 70, 30)
    income = st.number_input("💰 Annual Income", min_value=1000, step=1000)
    education = st.selectbox("🎓 Education Level", model.encoders["EducationLevel"].classes_)
    credit_score = st.slider("📈 Credit Score", 300, 850, 600)
    employment = st.selectbox("💼 Employment Status", model.encoders["EmploymentStatus"].classes_)

    submitted = st.form_submit_button("Predict Loan Approval")

# Prediction
if submitted:
    user_input = [age, income, education, credit_score, employment]
    prediction, prob = model.predict(user_input)

    st.markdown("### 🧠 Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Loan Approved with {prob:.2%} confidence.")
        st.balloons()
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {prob:.2%})")
        st.snow()
