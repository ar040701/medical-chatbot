# 🩺 Medical Chatbot

A Streamlit-based **Medical Chatbot** powered by **LangChain**, **Groq LLM**, **Hugging Face Embeddings**, and **FAISS vector search**.

The chatbot answers medical questions using a pre-built medical knowledge base and also allows users to upload a medical PDF for document-based question answering.

## 🔗 Live Demo

https://medical-chatbot-hxqvgx7ydqrjbs3rkhfxzv.streamlit.app/

## ✨ Features

- Medical question-answering chatbot
- Upload a medical PDF and ask questions from it
- Retrieval-Augmented Generation using FAISS
- Hugging Face sentence-transformer embeddings
- Groq LLM integration
- Chat history support
- Clear chat option
- Simple Streamlit web interface

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Web app interface |
| LangChain | RAG pipeline |
| Groq | Large language model |
| Hugging Face | Text embeddings |
| FAISS | Vector database |
| PyPDFLoader | PDF document loading |

## ⚙️ How It Works

```text
User question
    ↓
Retrieve relevant medical context from FAISS
    ↓
Send context + question to Groq LLM
    ↓
Generate medical response
```

If a PDF is uploaded, the app creates a temporary vector database from the PDF and answers questions based on that document.

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-chatbot.git
cd medical-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### 5. Run the app

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
medical-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── vector_store/
    └── db_faiss/
```

## ⚠️ Disclaimer

This chatbot is for **educational and informational purposes only**.  
It is not a replacement for professional medical advice, diagnosis, or treatment. Always consult a qualified doctor or healthcare professional for medical concerns.

## 👨‍💻 Author

Developed by **Ayush**
