#importing libraries
import json
import os
from huggingface_hub import InferenceClient
import gradio as gr
import time
from transformers import pipeline

import speech_recognition as sr



# import streamlit as st
# from streamlit_chat import message

my_db ={}
client = InferenceClient(
    # "mosaicml/mpt-7b-instruct"
    "mistralai/Mistral-7B-Instruct-v0.1"
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
    prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )
    # print("prompt",prompt)
    # print('history',history)
    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    my_db[prompt]=output
    # print(my_db)
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
    # print("output",output)
    return output   

def chatbot_audio(audio_data, history):
  # Recognize speech from audio data
  r = sr.Recognizer()
  with sr.AudioFile(audio_data) as source:
    audio = r.record(source)
  text = r.recognize_google(audio)

  # Generate chatbot response based on text
  response = f"You said: {text}"

  # Return text response
  return response



# def main():
#     st.set_page_config(
#         page_title="Your own ChatGPT",
#         page_icon="🤖"
#     )
    
#     st.header("Your own ChatGPT 🤖")
#     # user_input = st.text_input("Your message: ", key="user_input")
#     prompt = st.text_input("Enter your prompt:")
#     # print(prompt) 
    
#     message(prompt,is_user=True)
#     # message("hello")
    
#     ############################
    
    
#     history = []
    
#     # temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9)
#     # max_new_tokens = st.number_input("Max New Tokens", value=256)
#     # top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95)
#     # repetition_penalty = st.number_input("Repetition Penalty", value=1.0)
    
#     temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f")
#     max_new_tokens = st.slider("Max new tokens", min_value=0, max_value=1048, value=256, step=64)
#     top_p = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f")
#     repetition_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05, format="%.2f")
    
#     result = generate(prompt, history, temperature, max_new_tokens, top_p, repetition_penalty)
#     st.write(result)
#     print(result)
    


additional_inputs=[
    # gr.Slider(
    #     label="Temperature",
    #     value=0.9,
    #     minimum=0.0,
    #     maximum=1.0,
    #     step=0.05,
    #     interactive=True,
    #     info="Higher values produce more diverse outputs",
    # ),
    # gr.Slider(
    #     label="Max new tokens",
    #     value=256,
    #     minimum=0,
    #     maximum=1048, 
    #     step=64,
    #     interactive=True,
    #     info="The maximum numbers of new tokens",
    # ),
    # gr.Slider(
    #     label="Top-p (nucleus sampling)",
    #     value=0.90,
    #     minimum=0.0,
    #     maximum=1,
    #     step=0.05,
    #     interactive=True,
    #     info="Higher values sample more low-probability tokens",
    # ),
    # gr.Slider(
    #     label="Repetition penalty",
    #     value=1.2,
    #     minimum=1.0,
    #     maximum=2.0,
    #     step=0.05,
    #     interactive=True,
    #     info="Penalize repeated tokens",
    # ),
]


# gr.ChatInterface(
#     fn=generate,
#     chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
#     additional_inputs=additional_inputs,
#     title="""AgriChat"""
# ).launch(show_api=False)



interface = gr.Interface(
    fn=chatbot_audio,
    inputs=gr.Audio(sources=["microphone","upload"]),
    outputs="text",
    title="Speech-to-Text Chatbot",
)
interface.launch()


# if __name__ == '__main__':
#     main()