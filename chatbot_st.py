#importing libraries
import os
import time
import json
import tempfile
import streamlit as st
from streamlit_chat import message

import whisper

import requests

def save_conversation_to_file(conversation_history):
    """Append the conversation history to a JSON file based on today's date."""
    date_today = time.strftime("%Y-%m-%d")
    filename = f"conversation_{date_today}.json"
    filepath = os.path.join("./chat data", filename)  # Adjust the path as needed
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
    
    if os.path.exists(filepath):
        # File exists, read the current content and append
        with open(filepath, 'r') as f:
            existing_content = json.load(f)
        existing_content.extend(conversation_history)
        with open(filepath, 'w') as f:
            json.dump(existing_content, f, indent=4)
    else:
        # File does not exist, create a new one
        with open(filepath, 'w') as f:
            json.dump(conversation_history, f, indent=4)
    print(f"Conversation updated in {filepath}")


def apply_custom_css():
    background_image_url = "https://images.pexels.com/photos/289334/pexels-photo-289334.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1s"  # Replace this with the URL to your image
    
    st.markdown(
        f"""
        <style>
        /* This targets the main content area */
        .stApp {{
            background-image: url({background_image_url});
            background-size: cover;
        }}
        /* This ensures sidebar's background remains solid */
        .css-1d391kg {{
            background: #fff; /* or any color you want for the sidebar background */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def transcribe_audio(audio_file):
    model=whisper.load_model("base")
    result = model.transcribe(audio_file, fp16=False)
    return result["text"]



def transcribe_audio_or_use_text_input(audio_file, text_input=None):
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
            tmp_audio.write(audio_file.read())
            tmp_audio_path = tmp_audio.name
        transcribed_text = transcribe_audio(tmp_audio_path)
    else:
        transcribed_text = text_input
    return transcribed_text

def query(prompt):
    history_str = " ".join([f"User: {user_msg} , Assistant: {assistant_msg}" for user_msg, assistant_msg in st.session_state.conversation_history])

    headers = {"Authorization": f"Bearer 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"}
    API_URL = f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    payload = {
        "inputs": f"[INST] <<SYS>> Your job is to talk like a personal assistant for a farmer. Every reponse must sound like the same. Also do remember the previous conversation {history_str} and answer accordingly <<SYS>> User: {prompt} Assistant: [/INST] ",
                "parameters": {
                    "max_new_tokens":2048, 
                    "top_p":0.9, 
                    "temperature":0.7
                    }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def main():
    apply_custom_css()
    print("\nserver started....\n")
    # st.set_page_config(
    #     page_title="AgriChat",
    #     page_icon="🤖"
    # )
    st.header("AgriChat 🤖")
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


    message("Good Morning, How can i assist you today!")
    # Add a selectbox for choosing the input type
    with st.sidebar:
        input_type = st.selectbox("Select Input Type", ["Text", "Audio"])
    if input_type == "Text":
        # Text input
        with st.sidebar:
            text_input = st.text_input("Enter text to transcribe")
            transcribed_text = text_input
            if st.button("Query"):
                response = query(transcribed_text)
                # st.write(response)
                res = response[0]['generated_text'].split('[/INST]')[1]
                st.session_state.conversation_history.append((transcribed_text, res))
    elif input_type == "Audio":
        # Audio input
        with st.sidebar:
            audio_file = st.file_uploader("Upload audio file", type=["mp3","ogg", "wav"])
            if audio_file is not None:
                transcribed_text = transcribe_audio_or_use_text_input(audio_file)
                if st.button("Query"):
                    response = query(transcribed_text)
                    # st.write(response)
                    res = response[0]['generated_text'].split('[/INST]')[1]
                    st.session_state.conversation_history.append((transcribed_text, res))

    conversation_history = st.session_state.get('conversation_history', [])
    # print(conversation_history)
    save_conversation_to_file(conversation_history)

    for i, (user_agent, bot) in enumerate(conversation_history):
        message(user_agent, is_user=True, key=f"{i}_user")
        message(bot, key=f"{i}_ai")

if __name__ == "__main__":
    main()



