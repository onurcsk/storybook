import streamlit as st
import requests
from PIL import Image
from transformers import AutoProcessor, TFBlipForConditionalGeneration
import tensorflow as tf

st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# App title
st.title("Image Captioning App")

# Image upload feature
uploaded_files = st.file_uploader("Upload up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("You can only upload up to 5 images.")
    else:
        for uploaded_file in uploaded_files:
            # Open the image
            image = Image.open(uploaded_file)

            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image
            # text = "A picture of"
            # inputs = processor(images=image, text=text, return_tensors="tf")
            inputs = processor(images=image, return_tensors="tf")

            # Generate caption
            # with tf.device('/CPU:0'):  # Use CPU to avoid potential issues with GPU
            outputs = model.generate(**inputs)

            # Decode and display the caption
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            st.write(f"**Caption:** {caption}")
