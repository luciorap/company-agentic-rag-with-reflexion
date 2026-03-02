from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
#from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

#client = Client()

#prompt = client.pull_prompt("rlm/rag-prompt")

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | llm | StrOutputParser()


