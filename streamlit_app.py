#importing libraries
import json
import os
import time
from langchain.schema import (
    HumanMessage,
    AIMessage
)

import streamlit as st
from streamlit_chat import message

import requests


def query(payload):
    headers = {"Authorization": f"Bearer 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"}
    API_URL = f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    json_body = {
        "inputs": f"[INST] <<SYS>> Your job is to talk like a personal assistant for a farmer. Every reponse must sound like the same. <<SYS>> {payload} [/INST] ",
                "parameters": {"max_new_tokens":128, "top_p":0.9, "temperature":0.7}
    }
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # data = json.dumps(json_body)
    response = requests.post(API_URL, headers=headers, json=json_body)
    return response.json()



def main():
    print("\nserver started....")
    st.set_page_config(
        page_title="AgriChat",
        page_icon="🤖"
    )
    st.header("AgriChat 🤖")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    message("Good Morning, How can i assist you today!")
    with st.sidebar:
        prompt = st.text_input("Enter your prompt:")
        if prompt: 
            st.session_state.messages.append(HumanMessage(content=prompt))
            # message(prompt,is_user=True)
            with st.spinner("Thinking..."):
                data = query(prompt)
                res = data[0]['generated_text'].split('[/INST]')[1]
            st.session_state.messages.append(
                AIMessage(content=res))
            
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[0:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()
