import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def create_vector_store_from_pdf(pdf_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embedding)

# ✅ Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# ✅ Build RAG chain
def get_rag_chain(db):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
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
               
    uploaded_file = st.sidebar.file_uploader("Upload Medical PDF", type="pdf")
    
    if uploaded_file and uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large (max 5MB)")
        return
    
    # Show chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_input = st.chat_input("Ask your question...")
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        if uploaded_file:
            if "pdf_db" not in st.session_state:
                st.session_state.pdf_db = create_vector_store_from_pdf(uploaded_file.getvalue())
                st.sidebar.success("PDF processed successfully!")
    
            db = st.session_state.pdf_db

        else:
            db = load_vector_store()

        rag_chain = get_rag_chain(db)
        
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
            
            
            if any(x in response.lower() for x in [
                "i don't know",
                "not provided in the context",
                 "context does not mention"
                ]):
                llm = load_llm()
                response = llm.invoke(user_input).content

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()