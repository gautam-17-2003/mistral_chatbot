#importing libraries
import json
import os
from huggingface_hub import InferenceClient
import gradio as gr
import time

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
    prompt, history, temperature=1, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    # temperature = float(temperature)
    # if temperature < 1e-2:
    #     temperature = 1e-2
    # top_p = float(top_p)

     
    # print("prompt",prompt)
    # print('history',history)
    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
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
    # print("output",output)
    return output   




gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    # additional_inputs=additional_inputs,
    title="""AgriChat"""
    ).launch(show_api=False)