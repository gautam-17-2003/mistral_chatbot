import streamlit as st
import tempfile
import requests
import whisper
from streamlit_chat import message
# from transcribe_function import transcribe_audio  # Import your transcribe_audio function from your module


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

    headers = {"Authorization": f"Bearer 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"}
    API_URL = f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("prompt:",prompt)
    payload = {
        "inputs": f"[INST] <<SYS>> Your job is to talk like a personal assistant for a farmer. Every reponse must sound like the same. <<SYS>> User: {prompt} Assistant: [/INST] ",
                "parameters": {
                    "max_new_tokens":128, 
                    "top_p":0.9, 
                    "temperature":0.5
                    }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def main():
    
    print("\nserver started....\n")
    st.set_page_config(
        page_title="AgriChat",
        page_icon="🤖"
    )
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
    print(conversation_history)
    # store_db(conversation_history)

    for i, (user_agent, bot) in enumerate(conversation_history):
        message(user_agent, is_user=True, key=f"{i}_user")
        message(bot, key=f"{i}_ai")
    
    # input_text = st.text_input('enter text')
    # if input_text:
    #     data = query(input_text)
    #     st.write(data)

if __name__ == "__main__":
    main()
