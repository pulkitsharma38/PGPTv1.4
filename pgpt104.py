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

os.environ["OPENAI_API_KEY"] = apikey
llm = ChatOpenAI(temperature=0)


def main():
  st.title("Udyog Mitr AI")
  user_query = st.text_input("Search Product")
  t2 = time.time()
  vectorstore = get_data()
  t3 = time.time()
  print(user_query)
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
    t4 = time.time()
    result = conversation_chain({"question": prompt})
    t5 = time.time()
    response = st.write(result["answer"])
    print('Timing: get_data, retreive result', t3-t2, t5-t4)

if __name__ == "__main__":
  main()


# query = "compare specification, price & vendor details of sporty saftey shoes in a table. Put 1 specification in a row"
# result = conversation_chain.invoke({"question": query})
# answer = result["answer"]
# print(answer)