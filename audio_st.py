import streamlit as st
from audio_recorder_streamlit import audio_recorder

def main():
    audio_bytes = audio_recorder()
    if audio_bytes:
        input_audio = st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()