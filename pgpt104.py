####
# Chatgpt, custom prompt, cache vectorstore, ConversationalRetrievalChain
# pip install langchain langchain_openai langchain_community tiktoken apikey streamlit
####


from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tiktoken
import os
from apikey import apikey
import streamlit as st
import time
from dotenv import load_dotenv

@st.cache_resource
def get_data():
  loader = CSVLoader(file_path="./data/products240313.csv", encoding='latin1')
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=20,
     )
  texts = text_splitter.split_documents(data)
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(texts, embedding=embeddings)
  return vectorstore

load_dotenv()
llm = ChatOpenAI(temperature=0)


def main():
  st.title("Udyog मित्र AI")
  user_query = st.text_input("Search Product")
  
  vectorstore = get_data()
  
  if user_query:    
    
    prompt_template = PromptTemplate.from_template(
      "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that we currently don't have the product but we are continuously adding new products please check back in few days Thank you for kind cooperation, don't try to make up an answer. Compare specification, price, Product url , vendor name, vendor turnover, vendor expertise, vendor location and vendor contact of {product} in a table. Show each specification in a row"
    )
    prompt = prompt_template.format(product=user_query)
    
    
    
    memory = ConversationBufferMemory(
      memory_key='chat_history', return_messages=True, output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory,
      verbose=True
    )
    
    result = conversation_chain({"question": prompt})
    
    response = st.write(result["answer"])
    

if __name__ == "__main__":
  main()
