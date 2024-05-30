import streamlit as st
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
@st.cache_resource
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
    text1 = "Write me a story."

    if genre:
        text1 = f"Write me a {genre} story"
    if reader_age:
        text1 += f" suitable for {reader_age}-year-olds"
    if num_words:
        text1 += f" with {num_words} words"
    if num_characters:
        text1 += f" and {num_characters} characters."
    else:
        text1 += "."

    if character_names or character_genders:
        characters_info = " The main characters are "
        if character_names and character_genders:
            characters_info += ", ".join([f"{name} ({gender})" if gender else name for name, gender in zip(character_names, character_genders)])
        else:
            characters_info += ", ".join(character_names if character_names else character_genders)
        text1 += characters_info + "."

    text1 += " The story should be engaging and didactic. It should have a clear introduction, development, and a clear ending."
    if image_captions:
        text1 += " The following captions should be integrated in the story to contribute to the story development: " + ", ".join(image_captions)

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
st.markdown("<h2 style='text-align: center; color: #ff69b4;'>Create a Unique Story for Your Child!</h2>", unsafe_allow_html=True)

# Image upload and captioning
uploaded_files = st.file_uploader("Upload up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_captions = []

if 'cached_captions' not in st.session_state:
    st.session_state.cached_captions = {}

if 'story_history' not in st.session_state:
    st.session_state.story_history = []

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("You can only upload up to 5 images.")
    else:
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx]:
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

                # Display the image and caption
                st.image(image, caption=caption, use_column_width=True)

# User inputs under a dropdown
with st.expander("Story Settings"):
    genre = st.text_input("Story Genre or Theme (optional):", key="genre_input")
    num_words = st.slider("Number of Words (optional):", min_value=50, max_value=1000, step=50, key="num_words_slider")
    num_characters = st.slider("Number of Characters (optional):", min_value=1, max_value=10, step=1, key="num_characters_slider")
    reader_age = st.slider("Reader's Age (optional):", min_value=1, max_value=18, step=1, key="reader_age_slider")
    character_names = st.text_area("Character Names (comma-separated, optional):", key="character_names_area")
    character_genders = st.text_area("Character Genders (comma-separated, optional):", key="character_genders_area")

# Convert character names and genders to lists
character_names = [name.strip() for name in character_names.split(",") if name.strip()]
character_genders = [gender.strip() for gender in character_genders.split(",") if gender.strip()]

# Generate the story
if st.button("Generate the story!"):
    with st.spinner('Generating story...'):
        generated_story = generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders, image_captions)

    # Save the story and its details in the session state
    story_details = {
        "genre": genre,
        "num_words": num_words,
        "num_characters": num_characters,
        "reader_age": reader_age,
        "character_names": character_names,
        "character_genders": character_genders,
        "captions": image_captions,
        "story": generated_story
    }

    # Append the new story details to the history, limit to 5
    st.session_state.story_history.append(story_details)
    if len(st.session_state.story_history) > 5:
        st.session_state.story_history.pop(0)

    st.subheader("Generated Story:")
    st.write(generated_story)
    # Allow downloading the story
    story_filename = f"story_{hashlib.md5(generated_story.encode()).hexdigest()}.txt"
    st.download_button("Download Story", generated_story, file_name=story_filename)

# Display the story history
if st.session_state.story_history:
    with st.expander("View Story History"):
        tab_titles = [f"Story {i+1}" for i in range(len(st.session_state.story_history))]
        tabs = st.tabs(tab_titles)

        for idx, (tab, story_details) in enumerate(zip(tabs, st.session_state.story_history)):
            with tab:
                st.write(f"**Story Genre or Theme:** {story_details['genre']}")
                st.write(f"**Number of Words:** {story_details['num_words']}")
                st.write(f"**Number of Characters:** {story_details['num_characters']}")
                st.write(f"**Reader's Age:** {story_details['reader_age']}")
                st.write(f"**Character Names:** {', '.join(story_details['character_names'])}")
                st.write(f"**Character Genders:** {', '.join(story_details['character_genders'])}")
                st.write(f"**Captions:** {', '.join(story_details['captions'])}")
                st.write(f"**Story:** {story_details['story']}")
