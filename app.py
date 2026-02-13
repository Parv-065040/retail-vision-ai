import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. Page Configuration
st.set_page_config(
    page_title="RetailVision AI | Auto-Tagger",
    page_icon="🛍️",
    layout="wide"
)

# 2. Exotic UI CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .st-emotion-cache-1y4p8pa { padding-top: 2rem; }
    .upload-text { font-size: 20px; font-weight: bold; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# 3. Define the Clothing Categories
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 4. Load the CNN Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('vision_model.h5')

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model not found. Please ensure 'vision_model.h5' is uploaded to GitHub.")
    st.stop()

# 5. Sidebar Setup
with st.sidebar:
    st.title("🛍️ RetailVision AI")
    st.markdown("---")
    st.write("Automated visual cataloging tool for E-Commerce managers.")
    st.write("**Architecture:** Deep CNN")
    st.write("**Target ROI:** Reduce manual tagging hours by 94%.")
    st.markdown("---")
    st.caption("Upload a clothing item to see the AI automatically categorize it for the database.")

# 6. Main Dashboard Interface
st.title("⚡ E-Commerce Image Auto-Tagger")
st.write("Upload a raw product image. The neural network will extract visual features and assign the correct inventory tag.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<p class="upload-text">1. Upload Product Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a clothing image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Raw Upload', use_column_width=True)

with col2:
    st.markdown('<p class="upload-text">2. AI Analysis & Metadata</p>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner('CNN extracting spatial features...'):
            # Preprocess the image to match the neural network's expectations (28x28 grayscale)
            # We invert the colors because Fashion MNIST was trained on dark backgrounds
            img = image.convert('L')
            img = ImageOps.invert(img)
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Display Results
            st.success("Analysis Complete!")
            st.metric(label="Generated Inventory Tag", value=predicted_class, delta=f"{confidence:.2f}% Confidence")
            
            st.markdown("### Category Probability Matrix")
            # Create a nice visual bar chart of all probabilities
            import pandas as pd
            chart_data = pd.DataFrame(
                predictions[0],
                index=CLASS_NAMES,
                columns=['Probability']
            )
            st.bar_chart(chart_data)
    else:
        st.info("Awaiting image upload to begin analysis...")