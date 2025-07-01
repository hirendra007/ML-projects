# 🧠 Handwritten Digit Recognition (MNIST)

![MNIST Recognition Demo](https://i.imgur.com/Jf4h3Rl.gif)

A real-time digit classifier using TensorFlow and Streamlit that achieves **99.25% accuracy** on MNIST test data.

## 📁 Files

- `app.py` – Streamlit web application  
- `mnist_model.h5` – Pre-trained model  
- `model_trainer.py` – Python script to train the CNN model  
- `Handwritten Digit Recognition (MNIST).ipynb` – Training notebook (Colab-ready)

## 🚀 Quick Start

### Try in Colab  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i8IxYdGcwZ9k9LxhA3BTYw7OAx_TgYrJ?usp=sharing)

### Run Locally

```bash
pip install streamlit tensorflow opencv-python
streamlit run app.py
```

## 📊 Model Performance

### Training Metrics
```
157/157 - 3s 19ms/step - accuracy: 0.9898 - loss: 0.0759 
precision: 0.9899 - recall: 0.9898
```

### Final Test Results
| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 0.9925 |
| Precision         | 0.9926 |
| Recall            | 0.9925 |
| Inference Speed   | 39ms/step |

### Classification Report
```
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       980
           1       1.00      1.00      1.00      1135
           2       0.99      1.00      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.99      0.99      0.99       982
           5       1.00      0.97      0.99       892
           6       1.00      0.99      0.99       958
           7       0.99      1.00      0.99      1028
           8       1.00      0.99      1.00       974
           9       0.98      0.99      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

## 🛠 How It Works

- Draw a digit (0–9) on the canvas  
- Click the **"Predict"** button  
- Model predicts the digit and shows confidence scores

## 💡 Tips for Best Results

- Keep digits centered and large  
- Use brush stroke width between **12–15**  
- Avoid shaky/skewed strokes  
- Common mistakes:
  - 5 → 3 (open top)
  - 8 → 0 (too thin)
  - 9 → 4 (short tail)

## 🧠 Model Architecture

- Model: Custom CNN (3 Conv layers + BN + Dropout)  
- Input Shape: (28, 28, 1)  
- Activation: ReLU, Output: Softmax  
- Optimizer: Adam  
- Loss: categorical_crossentropy  
- Epochs: 10  
- Batch Size: 64  
- Test Accuracy: >99%  
- Saved as: `mnist_model.h5`

## 📝 License

MIT License – Free for academic and commercial use

> ✨ Built with TensorFlow, OpenCV, and Streamlit — by Hirendra
