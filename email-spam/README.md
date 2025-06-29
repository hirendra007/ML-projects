# 📧 Email Spam Detection using Naive Bayes

Spam emails waste time, clutter inboxes, and pose security risks. This project demonstrates how to build a lightweight and effective machine learning model to automatically detect spam messages using **Naive Bayes** and **TF-IDF vectorization**.

---

## 🔗 Google Colab Notebook

You can run the entire project directly in Google Colab:

👉 [Open in Colab](YOUR_GOOGLE_COLAB_LINK_HERE)

---

## ✨ Project Overview

| Feature | Description |
|--------|-------------|
| 📊 Dataset | spam-email-classification Dataset |
| 🧠 Model | Multinomial Naive Bayes |
| 🧼 Preprocessing | Lowercasing, punctuation removal, optional stopword removal |
| 🔤 Vectorization | TF-IDF |
| 🎯 Evaluation | Accuracy, Precision, Recall, F1-score |

---

## 📝 Dataset

- Source: [spam-email-classification Dataset](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification/data)
- Labels: `ham` (not spam), `spam` (junk)

---

## ✅ Model Performance

> 📸 *Add a screenshot of your classification report output below this line*

![Model Accuracy Screenshot](relative/path/to/your/screenshot.png)

---

## 🔍 Testing With Custom Messages

You can try your own text messages in real-time using the following function:

```python
def predict_message(message):
    message_clean = re.sub(r'[^a-zA-Z\s]', '', message.lower())
    message_vec = vectorizer.transform([message_clean])
    prediction = model.predict(message_vec)[0]
    return "SPAM" if prediction == 1 else "HAM"
