#importing libraries
import json
import os
from huggingface_hub import InferenceClient
import time

import streamlit as st
from streamlit_chat import message

my_db ={}
client = InferenceClient(
    # "mosaicml/mpt-7b-instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
    
def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
        my_db[user_prompt]=bot_response
    prompt += f"[INST] {message} [/INST]"
    
    return prompt

def generate(
    prompt, history, temperature=0.8, max_new_tokens=64, top_p=0.95, repetition_penalty=1.0,
):
    # temperature = float(temperature)
    # if temperature < 1e-2:
    #     temperature = 1e-2
    # top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=float(top_p),
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )
    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    # print(next(iter(stream)))
    print(stream)
    output = ""


    for response in stream:
        output += response.token.text
        yield output
        
    ###creating db
    my_db[prompt]=output
    os.chdir('./chat data')
    _file_name=""
    for  i in time.ctime().split(" "):
        _file_name += i
    file_name =""
    for i,name in enumerate(_file_name.split(":")):
        if i<=0:
            file_name+=name
        else:
            file_name+='_'+name
    # print(file_name)

    json_data = json.dumps(my_db, indent=4)  # `indent` for pretty formatting (optional)

    with open(f"{file_name}.json", "w") as json_file:
        json_file.write(json_data)
    os.chdir('C:\gautam\gpt cli')
    ###############
    
    print(output)
    return output   


def main():
    print("\nserver strarted.....")
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="🤖"
    )
    
    st.header("Your own ChatGPT 🤖")
    message("Good Morning, How can i assist you today!")
    with st.sidebar:
        prompt = st.text_input("Enter your prompt:")
    if prompt:
        message(prompt,is_user=True)
        history = []
        print(True)
        result = generate(prompt, history)
        print(result)
        st.write(result)   
    
    
    




if __name__ == '__main__':
    main()