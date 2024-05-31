import streamlit as st
from PIL import Image
import os
from io import BytesIO
import hashlib
import requests

API_HOST = os.environ['API_HOST']

# Streamlit app configuration
st.set_page_config(
    page_title="Image Captioning and Story Generation App",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Function to hash image
def hash_image(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = buffer.getvalue()
    return hashlib.md5(img_str).hexdigest()

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
                caption = requests.post(API_HOST + "generate_caption/", files={"file": uploaded_file.getvalue()}).json()["caption"]
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
    data={"genre": genre, "num_words" : num_words, "num_characters" : num_characters, "reader_age" : reader_age, "character_names" : character_names, "character_genders" : character_genders, "image_captions" : image_captions}
    generated_story = requests.post(API_HOST + "generate_story/", data=data).json()["story"]
    
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
