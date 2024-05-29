import streamlit as st
import requests
from PIL import Image
from transformers import AutoProcessor, TFBlipForConditionalGeneration
import tensorflow as tf
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import os
from io import BytesIO
import hashlib

# Streamlit app configuration
st.set_page_config(
    page_title="Image Captioning and Story Generation App",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize Vertex AI
aiplatform.init(project=os.environ["GOOGLE_PROJECT_ID"], location=os.environ["GOOGLE_PROJECT_REGION"])

# Load the image captioning model and processor
@st.cache(allow_output_mutation=True)
def load_captioning_model():
    model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_captioning_model()

# Function to generate image captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="tf")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to hash image
def hash_image(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = buffer.getvalue()
    return hashlib.md5(img_str).hexdigest()

# Function to generate the story
def generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders, image_captions):
    text1 = f"Write me a {genre} story suitable for {reader_age}-year-olds. The story should have {num_words} words and {num_characters} characters."
    for name, gender in zip(character_names, character_genders):
        text1 += f" The main character is {name}, a {gender}."
    text1 += " The story should be engaging and didactic. It should have a clear introduction, development, and a clear ending."
    text1 += " The following captions should be integrated in the story to contribute in the story development: " + ", ".join(image_captions)

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Generate story prompt
    model = GenerativeModel("gemini-1.5-flash-001")
    responses = model.generate_content([text1], generation_config=generation_config, safety_settings=safety_settings, stream=True)

    generated_story = ""
    for response in responses:
        generated_story += response.text

    return generated_story

# User inputs for story generation
st.title("Image Captioning and Story Generation App")
genre = st.text_input("Enter the genre of the story or a theme description:")
num_words = st.number_input("Enter the number of words in the story:", min_value=1, step=1)
num_characters = st.number_input("Enter the number of characters in the story:", min_value=1, step=1)
reader_age = st.number_input("Enter the reader's age:", min_value=1, step=1)
character_names = st.text_area("Enter the name(s) of the character(s), separated by commas:")
character_genders = st.text_area("Enter the gender(s) of the character(s), separated by commas:")

# Convert character names and genders to lists
character_names = [name.strip() for name in character_names.split(",")]
character_genders = [gender.strip() for gender in character_genders.split(",")]

# Image upload and captioning
uploaded_files = st.file_uploader("Upload up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_captions = []

if 'cached_captions' not in st.session_state:
    st.session_state.cached_captions = {}

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("You can only upload up to 5 images.")
    else:
        for uploaded_file in uploaded_files:
            # Open the image
            image = Image.open(uploaded_file)
            # Hash the image
            image_hash = hash_image(image)
            # Check if the image has been processed before
            if image_hash in st.session_state.cached_captions:
                caption = st.session_state.cached_captions[image_hash]
            else:
                # Generate and cache the caption
                caption = generate_caption(image)
                st.session_state.cached_captions[image_hash] = caption
            image_captions.append(caption)

        with st.expander("View uploaded images and captions"):
            for uploaded_file, caption in zip(uploaded_files, image_captions):
                # Open and display the image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                # Display the caption
                st.write(f"**Caption:** {caption}")

# Generate the story
if st.button("Generate the story!"):
    if not genre or not num_words or not num_characters or not reader_age or not character_names or not character_genders:
        st.warning("Please fill in all the fields.")
    else:
        generated_story = generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders, image_captions)
        st.subheader("Generated Story:")
        st.write(generated_story)
        