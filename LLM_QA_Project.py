from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import streamlit as st
import pickle

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
api_key="AIzaSyDS5eZnM4jL3kFXj0kzEK77Tq32ZnQEckI"

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=api_key, temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index.pkl"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path=r'C:\Users\SKAN\Downloads\codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    with open(vectordb_file_path, "wb") as f:
        pickle.dump(vectordb, f)

def get_qa_chain():
    if os.path.exists(vectordb_file_path):
        with open(vectordb_file_path, "rb") as f:
            vectorstore = pickle.load(f)
    # Create a retriever for querying the vector database
    retriever = vectorstore.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()

st.title("Codebasics Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])