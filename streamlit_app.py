import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the saved model
@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5")  # Adjust path if needed

model = load_trained_model()

# Define the classes
classes = ['soil', 'broadleaf', 'grass', 'soybean']

def predict_image_from_path(image):
    """
    Predicts the class of a given image using the loaded model.

    Args:
        image (PIL.Image): The input image.

    Returns:
        tuple: Predicted class (str) and class probabilities (dict).
    """
    try:
        image = image.convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)[0]
        class_probabilities = {classes[i]: float(predictions[i]) for i in range(len(classes))}

        predicted_class_index = np.argmax(predictions)
        predicted_class = classes[predicted_class_index]

        return predicted_class, class_probabilities

    except Exception as e:
        return f"Error during prediction: {e}", None

# --- Streamlit UI ---
st.title("🌿 Real-Time Weed Detection")
st.write("Upload an image to classify it as Soil, Broadleaf, Grass, or Soybean.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")
        predicted_label, probabilities = predict_image_from_path(image)

        if probabilities:
            st.success(f"**Predicted Class:** {predicted_label.capitalize()}")
            st.write("### Class Probabilities:")
            st.json(probabilities)

            # Display bar chart
            st.bar_chart(probabilities)
        else:
            st.error(predicted_label)

    except Exception as e:
        st.error(f"Failed to process image: {e}")
