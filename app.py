import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.vectorstores import FAISS
from langchain.embeddings import TensorflowHubEmbeddings

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.STOP_SEQUENCES: ["\n"],
    GenParams.REPETITION_PENALTY:1
    }


load_dotenv()
project_id = os.getenv("PROJECT_ID", None)
credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": os.getenv("API_KEY", None)
        }    
#this cell should never fail, and will produce no output
import requests

def getBearer(apikey):
    form = {'apikey': apikey, 'grant_type': "urn:ibm:params:oauth:grant-type:apikey"}
    print("About to create bearer")
#    print(form)
    response = requests.post("https://iam.cloud.ibm.com/oidc/token", data = form)
    if response.status_code != 200:
        print("Bad response code retrieving token")
        raise Exception("Failed to get token, invalid status")
    json = response.json()
    if not json:
        print("Invalid/no JSON retrieving token")
        raise Exception("Failed to get token, invalid response")
    print("Bearer retrieved")
    return json.get("access_token")

credentials["token"] = getBearer(credentials["apikey"])
from ibm_watson_machine_learning.foundation_models import Model
model_id = ModelTypes.LLAMA_2_70B_CHAT

# Initialize the Watsonx foundation model
llama_model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id)


# Function to get text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text += " ".join(page.extract_text() for page in pdf_reader.pages)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    #embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings() 
    embeddings  = TensorflowHubEmbeddings(model_url=url)   
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm=llama_model.to_langchain()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display responses
def handle_user_input(user_question):
    parameters = {"instruction": "Answer the following question using only information from the article. If there is no good answer in the article, say I don't know"}
    prompt={"question": user_question}
    response = st.session_state.conversation(prompt, {"Answer the following question using only information from the article. If there is no good answer in the article, say I don't know"})
    st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function
def main():
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with PDFs documents :books:")
    user_question = st.text_input("Upload your documents and ask questions:")        
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if not pdf_docs:
            st.write('Please add a document')
        else:     
            if st.button("Process"):
                with st.spinner("Processing"):
                    # Get PDF text and split into chunks
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    # Create vector store and conversation chain
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Document loaded")
                    st.session_state.conversation = get_conversation_chain(vectorstore)                    
    if user_question and st.session_state.conversation is not None:
            handle_user_input(user_question)
        
            
if __name__ == '__main__':
    main()
