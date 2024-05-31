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

# User inputs under a dropdown
with st.expander("Story Settings"):
    tab_titles = ["Story Genre or Theme", "Number of Words", "Number of Characters", "Reader's Age", "Character Names", "Character Genders"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        genre = st.text_input("Enter the story genre or theme description (optional):")
    with tabs[1]:
        num_words = st.number_input("Enter the number of words in the story (optional):", min_value=1, step=1)
    with tabs[2]:
        num_characters = st.number_input("Enter the number of characters in the story (optional):", min_value=1, step=1)
    with tabs[3]:
        reader_age = st.number_input("Enter the reader's age (optional):", min_value=1, step=1)
    with tabs[4]:
        character_names = st.text_area("Enter the name(s) of the character(s), separated by commas (optional):")
    with tabs[5]:
        character_genders = st.text_area("Enter the gender(s) of the character(s), separated by commas (optional):")

# Convert character names and genders to lists
character_names = [name.strip() for name in character_names.split(",") if name.strip()]
character_genders = [gender.strip() for gender in character_genders.split(",") if gender.strip()]

# Generate the story
if st.button("Generate the story!"):
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
