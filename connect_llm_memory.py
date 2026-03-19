from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",   # fast & good
        temperature=0.7
    )

prompt = ChatPromptTemplate.from_template("""
Use the context to answer the question.
If you don't know, say you don't know.

Context: {context}
Question: {question}

Answer:
""")

DB_FAISS_PATH = "vector_store/db_faiss"

#Load Database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

#Create QA chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = load_llm()

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

#Invoke with a query
user_query = input("Write Query here: ")
response = rag_chain.invoke(user_query)
print("Answer: ", response)
