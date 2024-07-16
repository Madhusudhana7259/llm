# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


# os.environ['']=os.getenv("")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant"),
        ("user","Question:{question}")
    ]
)


st.title("LANGCHAIN")
ip_text = st.text_input("Search the topic")

llm = Ollama(model="gemma2")
out_parser = StrOutputParser()
chain = prompt|llm|out_parser

if ip_text:
    st.write(chain.invoke({"question":ip_text}))