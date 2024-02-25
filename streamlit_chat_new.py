#importing libraries
import json
import os
import csv

import streamlit as st
from streamlit_chat import message

import requests


def query(payload):
    history_str = " ".join([f"User: {user_msg} , Assistant: {assistant_msg}" for user_msg, assistant_msg in st.session_state.conversation_history])

    headers = {"Authorization": f"Bearer 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"}
    API_URL = f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    

    json_body = {
                # "messages": [
                #     {
                #         "role": "system",
                #         "content": f"You are a helpful assistant in field of farming. Also do remember the previous chat conversation as {history_str} "
                #     },
                #     {
                #         "role": "user",
                #         "content": f"[INST] User: {payload} Assistant: [/INST] "
                #     }
                # ],
                "inputs": f"[INST] <<SYS>> Your job is to talk like a personal assistant for a farmer. Every reponse must sound like the same. Also do remember the previous conversation {history_str} and answer accordingly <<SYS>> User: {payload} Assistant: [/INST] ",
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 16,   
                "stream": True,
                # "safe_prompt": false,
                # "random_seed": null
                }

    response = requests.post(API_URL, headers=headers, json=json_body)
    return response.json()


def store_db(chat_data):
    fields = ['user','bot']
    filename = 'data_set.csv'

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(chat_data)


def main():
    print("\nserver started....\n")
    st.set_page_config(
        page_title="AgriChat",
        page_icon="🤖"
    )
    st.header("AgriChat 🤖")
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []


    message("Good Morning, How can i assist you today!")
    with st.sidebar:
        prompt = st.text_input("Enter your prompt:")
        # st.session_state.user_txt.append(prompt)
    if prompt:
        # history_str = " ".join([f"User: {user_msg} , Assistant: {assistant_msg}" for user_msg, assistant_msg in st.session_state.conversation_history])
        with st.spinner("Thinking...."):
            data = query(prompt)
            # print(data)
            res = data["choices"][0]["message"]["content"]
            print(res)
        st.session_state.conversation_history.append((prompt, res))

    conversation_history = st.session_state.get('conversation_history', [])
    print(conversation_history)
    store_db(conversation_history)
    for i, (user_agent, bot) in enumerate(conversation_history):
        message(user_agent, is_user=True, key=f"{i}_user")
        message(bot, key=f"{i}_ai")



if __name__ == '__main__':
    main()