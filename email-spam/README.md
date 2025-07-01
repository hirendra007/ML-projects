# 📧 Email Spam Detection using Naive Bayes

Spam emails waste time, clutter inboxes, and pose security risks. This project demonstrates how to build a lightweight and effective machine learning model to automatically detect spam messages using **Naive Bayes** and **TF-IDF vectorization**.

---

## 🔗 Google Colab Notebook

You can run the entire project directly in Google Colab:

👉 [Open in Colab](https://colab.research.google.com/drive/1jhXALjpn6X30B5mx9H_kJa5yTUcjdLLO?usp=sharing)

---

## ✨ Project Overview

| Feature        | Description                                            |
|----------------|--------------------------------------------------------|
| 📊 Dataset     | [Spam Email Classification Dataset](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification/data) |
| 🧠 Model       | Multinomial Naive Bayes                                |
| 🧼 Preprocessing | Lowercasing, punctuation removal, optional stopword removal |
| 🔤 Vectorization | TF-IDF (Term Frequency - Inverse Document Frequency) |
| 🎯 Evaluation  | Accuracy, Precision, Recall, F1-score                  |

---

## 📝 Dataset

- Source: [Spam Email Classification Dataset on Kaggle](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification/data)
- Labels: `ham` (not spam), `spam` (junk)

---

## ✅ Model Performance

```text
✅ Accuracy: 0.9677

📊 Classification Report:
              precision    recall  f1-score   support

           0       0.96      1.00      0.98       966
           1       1.00      0.76      0.86       149

    accuracy                           0.97      1115
   macro avg       0.98      0.88      0.92      1115
weighted avg       0.97      0.97      0.97      1115

🔍 Confusion Matrix:
[[966   0]
 [ 36 113]]