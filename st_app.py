import streamlit as st
from huggingface_hub import InferenceApi
import os
import datetime

# Initialize the Huggingface Inference API client
client = InferenceApi("mistralai/Mistral-7B-Instruct-v0.1")

# Define the chat history folder path
chat_history_folder = "chat_history"
# Ensure the chat history folder exists
if not os.path.exists(chat_history_folder):
    os.makedirs(chat_history_folder)

def get_todays_chat_history_file_path():
    today = datetime.date.today()
    filename = f"chat_history_{today}.txt"
    return os.path.join(chat_history_folder, filename)

def save_chat_to_history(user_input, bot_response):
    file_path = get_todays_chat_history_file_path()
    with open(file_path, "a") as file:
        file.write(f"User: {user_input}\n")
        file.write(f"Bot: {bot_response}\n\n")

def load_todays_chat_history():
    file_path = get_todays_chat_history_file_path()
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            chat_history = file.read().strip().split("\n\n")
            for exchange in chat_history:
                user_line, bot_line = exchange.split("\n")
                user_input = user_line[len("User: "):]
                bot_response = bot_line[len("Bot: "):]
                st.session_state['chat_history'].append((user_input, bot_response))

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    load_todays_chat_history()

def generate_response(user_input):
    prompt = f"{user_input}\n\n###\n\n"
    for user_input, bot_response in reversed(st.session_state['chat_history'][-5:]):
        prompt = f"{user_input} {bot_response} " + prompt

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_length": 512,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        },
    }

    response = client(payload)
    bot_response = response[0]["generated_text"].strip()

    st.session_state['chat_history'].append((user_input, bot_response))
    save_chat_to_history(user_input, bot_response)

    return bot_response

def apply_custom_css():
    background_image_url = "https://www.logineko.com/wp-content/uploads/2023/10/organic-cereals-768x513.webp"
    
    st.markdown(f"""
        <style>
        /* Full page background */
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Adjust transparency and style of containers for better readability */
        .stTextInput>div>div>input, .stButton>button {{
            border-radius: 20px;
            padding: 10px;
            margin: 5px 0;
        }}

        .stButton>button {{
            border: 1px solid #4CAF50;
            color: white;
            background-color: #4CAF50;
        }}

        .css-1d391kg {{
            padding-top: 0;
            padding-bottom: 0;
        }}

        /* Custom styles for markdown containers */
        .stMarkdownContainer {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #ddd;
        }}

        /* Custom title color and styles */
        h1 {{
            color: #3e7b5a;
        }}
        </style>
    """, unsafe_allow_html=True)

# Call this function in your main() to apply the styles



def main():
    apply_custom_css()
    st.title("AgriChat: Your Agricultural Assistant")
    
    chat_area = st.empty()
    
    user_input = st.text_input("Type your message here:","", key="user_input")

    if st.button("Send") and user_input.strip():
        bot_response = generate_response(user_input.strip())
        st.session_state.user_input = ""  
        display_chat_history(chat_area, bot_response)

def display_chat_history(chat_area, bot_response=None):
    with chat_area.container():
        for user_input, bot_response in st.session_state['chat_history']:
            st.container().markdown(f"**You**: {user_input}")
            st.container().markdown(f"**AgriBot**: {bot_response}")

if __name__ == "__main__":
    main()
    
    