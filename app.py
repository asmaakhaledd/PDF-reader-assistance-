import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import sqlite3
import pickle
import redis

#djncnjkfnjkvdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.redis import Redis
from langchain.vectorstores import Redis as VectorStoreRedis 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

file_uploaded = False


redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_client = redis.StrictRedis.from_url(redis_url)
PDF_KEY = "uploaded_pdf_content"
LOG_KEY = "question_log"
LOGIN_KEY = "login_data"


def save_pdf_content_to_redis(pdf_text):
    pdf_key = f"{PDF_KEY}_{len(redis_client.keys(PDF_KEY + '*')) + 1}"
    redis_client.append(pdf_key, pickle.dumps(pdf_text))
    return pdf_key


def retrieve_pdf_content_from_redis(pattern):
    pdf_contents = []
    keys = redis_client.keys(pattern)
    for key in keys:
        pdf_content = redis_client.get(key)
        if pdf_content:
            pdf_contents.append(pickle.loads(pdf_content))
    return pdf_contents


def save_log_to_redis(question, answer):
    log_entry = " ".join(["Question:", question, "Answer:", answer])
    redis_client.append(LOG_KEY, log_entry)
    
    
def pdf_reading(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_pdf_content(pdf):
    text = pdf_reading(pdf)
    pdf_key =save_pdf_content_to_redis(text)
    text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
    chunks = text_splitter.split_text(text)
    return pdf_key


def main():
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # if openai_api_key is None or openai_api_key == "":
    #     st.write("OPENAI_API_KEY is not set")
    #     exit(1)
    # else:
    #     st.write("OPENAI_API_KEY is set")
        
        
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # log in 
    st.sidebar.title("Login!")
    username = st.sidebar.text_input("Username:")
    password = st.sidebar.text_input("Password:", type="password")
    
    logged_in = False
    if st.sidebar.button("Login"):
        stored_login_data = redis_client.get(LOGIN_KEY)
        if stored_login_data:
            stored_username_password_pairs = stored_login_data.decode('utf-8').split("Username: ")[1:]            
            authenticated = False    
            for pair in stored_username_password_pairs: 
                stored_username, stored_password = pair.split(", Password: ")      
                if username == stored_username.strip() and password == stored_password.strip():
                    authenticated = True 
                    break   
            if authenticated:
                logged_in = True
                st.sidebar.success("Login successful!")
            else:
                st.sidebar.error("Invalid credentials. Please try again.")
                logged_in = False
    
    if "login_state" not in st.session_state :
        st.session_state.login_state = False
    
    if logged_in or st.session_state.login_state:
        st.session_state.login_state = True

        st.sidebar.empty()
    
        # upload file
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        # extract the text
        if pdf is not None:
                pdf_key = process_pdf_content(pdf)
                st.write("PDF Content Stored in Redis" ,pdf_key)
                # create embeddings
                embeddings = OpenAIEmbeddings()
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                                
                user_question = st.text_input("Ask a question about your PDF:")
                if user_question:
                    pattern = f"{PDF_KEY}_*"
                    pdf_contents = retrieve_pdf_content_from_redis(pattern)
                    knowledge_base = Redis.from_texts(pdf_contents, embeddings, redis_url=redis_url)
                    docs = knowledge_base.similarity_search(user_question) 
                    response = chain.run(input_documents=docs, question=user_question)
                    save_log_to_redis(user_question, response)
                    st.write(response)
                
                    
                with get_openai_callback() as cb:
                        print(cb)
            


if __name__ == '__main__':
    main()