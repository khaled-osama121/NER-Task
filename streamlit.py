import streamlit as st
import requests

# URL of your Flask API endpoint
API_URL = "http://localhost:8000/get_entities"  # Replace with your Flask API URL

# Function to perform NER by calling the Flask API
def extract_entities(text):
    response = requests.post(API_URL, json={"text": text})
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error: Unable to get entities from API")
        return []


def highlight_entities(text, entities):
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            text = text.replace(entity, f'<mark>{entity}</mark>')
    return text


# Streamlit app
st.title("Named Entity Recognition Demo")
st.write("Enter text to extract named entities:")

# Input text
user_input = st.text_area("Text", "Type your text here...")
if st.button("Extract Entities"):
    if user_input:
        entities = extract_entities(user_input)['entities']
        st.write("Extracted Entities:")
        for entity_type, entities in entities.items():
            st.write(f"**{entity_type}**:")
            for entity in entities:
                st.write(f"- {entity}")
    else:
        st.write("Please enter some text.")

