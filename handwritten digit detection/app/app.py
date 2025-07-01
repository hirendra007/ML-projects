import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set up the app
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🧠 MNIST Digit Recognizer")
st.markdown("Draw a digit (0-9) below and the AI will predict it!")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

# Canvas settings
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 12)
drawing_mode = st.sidebar.selectbox("Drawing tool:", ["freedraw", "transform"])

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent background
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",  # White drawing color
    background_color="#000000",  # Black canvas
    width=280,
    height=280,
    drawing_mode=drawing_mode,
    key="canvas"
)

# Preprocessing function to match model training
def preprocess_image(image_data):
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    # Invert colors (model expects white digit on black background)
    image_data = 255 - image_data
    
    # Resize to 28x28 (original MNIST size)
    image_data = cv2.resize(image_data, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Convert to 3 channels and resize to 128x128 for MobileNetV2
    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
    image_data = cv2.resize(image_data, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    image_data = image_data.astype("float32") / 255.0
    
    return image_data.reshape(1, 128, 128, 3), image_data

# Prediction button
if st.button("🎯 Predict Digit"):
    if canvas_result.image_data is not None:
        with st.spinner("Analyzing your drawing..."):
            # Preprocess the drawn image
            input_array, processed_img = preprocess_image(canvas_result.image_data)
            
            # Show the processed image
            st.subheader("What the model sees:")
            st.image(processed_img, width=150, clamp=True)
            
            # Make prediction
            predictions = model.predict(input_array)[0]
            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Display results
            st.success(f"**Predicted Digit:** {predicted_digit} (Confidence: {confidence:.1%})")
            
            # Show confidence scores
            st.subheader("Prediction Confidence:")
            probs = {str(i): float(predictions[i]) for i in range(10)}
            st.bar_chart(probs)
            
            # Add some fun
            if confidence > 0.95:
                st.balloons()
    else:
        st.warning("Please draw a digit first!")

# Add clear canvas button
if st.button("🧹 Clear Canvas"):
    st.experimental_rerun()

# Model info in sidebar
st.sidebar.header("Model Information")
st.sidebar.markdown("""
- **Architecture:** MobileNetV2
- **Input:** 128×128 RGB images
- **Training Accuracy:** ~99.5%
- **Preprocessing:**
  - White digit on black background
  - Resized to 128×128
  - Normalized to [0,1]
""")

# Tips for better results
st.sidebar.header("Drawing Tips")
st.sidebar.markdown("""
1. Draw in the center of the canvas
2. Make your digit large and clear
3. Use thicker strokes (adjust slider)
4. Avoid very light/dim drawings
5. The model works best with single digits
""")