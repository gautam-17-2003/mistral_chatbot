from dotenv import load_dotenv
import gradio as gr
import io
# import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms import HuggingFaceHub
from gtts import gTTS

# from InstructorEmbedding import INSTRUCTOR


# def get_pdf_from_url(url):
#     headers = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}
#     response = requests.get(url=url, headers=headers, timeout=120)
#     # Save the response content to a file
#     with open('document.pdf', 'wb') as file:
#         file.write(response.content)
#     return 'document.pdf'


def questiondocument(user_question, url):
    load_dotenv()
    document = 'document.pdf'
    pdf_reader = PdfReader(document)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
        
    # chunking
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text)

    # embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
      
    docs = knowledge_base.similarity_search(user_question)
        
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":2048})
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
      response = chain.run(input_documents=docs, question=user_question)
    audio = gTTS(response)
    audio.save("output.mp3")

    cost = cb
    return response, cost, "output.mp3"

demo = gr.Interface(
    fn=questiondocument,
    inputs=["text", "text"],
    outputs=["text","text","audio"],
)
demo.launch(share=True)
