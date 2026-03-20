import streamlit as st
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vector_store/db_faiss"

# ✅ Cache vector DB
@st.cache_resource
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# ✅ Load LLM
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

# ✅ Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Answer the question using the context if it is relevant.
If the context is incomplete or does not answer the question, ignore it and use your own knowledge.

Never say "I don't know" just because the context is missing.

Provide a clear, helpful, and safe medical response.
Always mention that the user should consult a doctor for medical advice.

Context:
{context}

Question:
{question}

Answer:
""")

# ✅ Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# ✅ Build RAG chain
@st.cache_resource
def get_rag_chain():
    db = load_vector_store()
    retriever = db.as_retriever(search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10})
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
    return rag_chain

# ✅ Streamlit UI
def main():
    st.title("🩺 Medical Chatbot")
    st.write("Ask me anything about medical topics.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_input = st.chat_input("Ask your question...")
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        
    

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        rag_chain = get_rag_chain()
        
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
            
            
            if "i don't know" in response.lower():
                llm = load_llm()
                response = llm.invoke(user_input).content

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()