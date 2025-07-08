# 🏦 Loan Eligibility Predictor

This project predicts whether a bank loan application will be **approved or not** using machine learning. It uses a combination of **Logistic Regression** and **Random Forest**, and is available as a **Streamlit web app**.

---

## 📊 Problem Statement

Banks want to predict whether a loan applicant is eligible for approval based on their personal and financial details.

---

## 🎯 Objective

Build a classification model that predicts loan approval based on:
- Age
- Income
- Education Level
- Credit Score
- Employment Status

---

## 🧠 ML Models Used

- ✅ Logistic Regression  
- ✅ Random Forest  
- ✅ Combined using soft voting for better accuracy

---

## 📁 Dataset

Dataset includes:
- `Age`, `AnnualIncome`, `EducationLevel`, `CreditScore`, `EmploymentStatus`, and `LoanApproved`

📥 **Download from Kaggle**:  
[👉 Loan Dataset on Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)

---

## 📓 Google Colab Notebook

Train the models and export the combined `.pkl` model for deployment.  
🔗 [Open in Google Colab](https://colab.research.google.com/drive/1bOmPwshSsbiLw_lkBm2dd13LTmBei3Bo?usp=sharing)

---

## 🖥️ Web App with Streamlit

### 💻 Run Locally

```bash
git clone https://github.com/hirendra007/ML/loan-eligibility-app.git
cd loan-eligibility-app/app
pip install -r requirements.txt
streamlit run app.py
