from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load Raw pdfs
dataPath = "data/"

def load_pdfs(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdfs(dataPath)
# print(f"Loaded {len(documents)} documents.")

# 2. Extract text from pdfs
def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
# print(f"Created {len(text_chunks)} text chunks.")
# print(f"Example chunk: {text_chunks[0]}")

# 3. Create embeddings from extracted text
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()
# 4. Store embeddings in a faiss
FAISS_PATH = "vector_store/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(FAISS_PATH)