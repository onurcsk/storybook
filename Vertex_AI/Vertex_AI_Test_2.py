import streamlit as st
from google.cloud import aiplatform
from vertexai import preview
# import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models


st.set_page_config(
    page_title="Story Generation App",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="auto"
)


# Initialize Vertex AI
aiplatform.init(project="tidal-mote-419711", location="europe-west1")

# Function to generate story
def generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders): # image_captions for later
    text1 = f"Write me a {genre} story suitable for {reader_age}-year-olds. The story should have {num_words} words and {num_characters} characters."
    for name, gender in zip(character_names, character_genders):
        text1 += f" The main character is {name}, a {gender}."
    text1 += " The story should be engaging and didactic. It should have a clear introduction, development, and a clear ending."
    
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

# User inputs
genre = st.text_input("Enter the genre of the story or a theme description:")
num_words = st.number_input("Enter the number of words in the story:", min_value=1, step=1)
num_characters = st.number_input("Enter the number of characters in the story:", min_value=1, step=1)
reader_age = st.number_input("Enter the reader's age:", min_value=1, step=1)
character_names = st.text_area("Enter the name(s) of the character(s), separated by commas:")
character_genders = st.text_area("Enter the gender(s) of the character(s), separated by commas:")
# Convert character names and genders to lists
character_names = [name.strip() for name in character_names.split(",")]
character_genders = [gender.strip() for gender in character_genders.split(",")]

# Image captioning
# uploaded_files = st.file_uploader("Upload up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# if uploaded_files:
#     # Process uploaded images and get captions
#     image_captions = []
#     for uploaded_file in uploaded_files:
#         # Process image and get caption
#         # This part should integrate with the image captioning model
#         # You can add this functionality here
    
#     # Generate story based on user inputs and image captions
#         generated_story = generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders, image_captions)
    
#     # Display generated story
#     st.subheader("Generated Story:")
#     st.write(generated_story)

if st.button("Generate the story!"):
    st.write(generate_story(genre, num_words, num_characters, reader_age, character_names, character_genders))