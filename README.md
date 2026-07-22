# 🩺 MedBot - AI Medical Chatbot using RAG
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-blueviolet)
![Groq](https://img.shields.io/badge/Groq-LLM-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 About the Project

MedBot is an AI medical chatbot that answers health-related questions using information from a medical book.

Instead of answering only from the AI model's general knowledge, the chatbot first searches the medical book for the most relevant information. It then uses that information to generate an answer. This approach is called Retrieval-Augmented Generation (RAG).

I built this project to learn how Large Language Models (LLMs), LangChain, Vector Databases, and Prompt Engineering work together in a real application.

## 🚀 Live Demo

🌐 **Try MedBot:** [medbot](https://medbott.streamlit.app/)

📂 **GitHub Repository:** https://github.com/OmkarSahu0375/medbot

## 📷 Screenshot

### Home Screen

<img width="959" height="434" alt="image" src="https://github.com/user-attachments/assets/cf45d59d-6f1d-40ec-9359-1246ef2cfbc7" />


### Chat Example

<img width="959" height="435" alt="image" src="https://github.com/user-attachments/assets/e1376f37-4817-48a0-b3c5-db13d9793a14" />

## ✨ Features

- Answer medical questions using a medical knowledge base
- Retrieve relevant information before generating an answer (RAG)
- Semantic search using Pinecone Vector Database
- Stream responses in real time
- Remember previous messages during the conversation
- Simple and clean Streamlit interface
- Faster loading using Streamlit caching

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | GPT-OSS-20B (Groq) |
| AI Framework | LangChain |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | Pinecone |
| Document Loader | PyPDF |

## 🏗️ System Architecture

The chatbot follows these steps to answer a user's question:

1. Medical PDF files are loaded and split into smaller chunks.
2. Each chunk is converted into embeddings using the MiniLM embedding model.
3. The embeddings are stored in Pinecone Vector Database.
4. When a user asks a question, the question is converted into an embedding.
5. Pinecone searches for the most relevant medical information.
6. The retrieved information, along with the user's question and conversation history, is sent to the LLM.
7. The LLM generates the final answer and streams it back to the user in real time.

## 📁 Project Structure

```text
medbot/
│
├── data/
│   └── medical-book.pdf
│
├── src/
│   ├── helper.py
│   └── prompt.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env
```

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/OmkarSahu0375/medbot.git
```

### Go to the project folder

```bash
cd medbot
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install the required packages

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

## 🔑 Environment Variables

Create a `.env` file in the project root and add the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | API key for Pinecone Vector Database |
| `GROQ_API_KEY` | API key for Groq LLM |

## ▶️ Usage

Run the application:

```bash
uv run streamlit run streamlit_app.py
```

After the server starts, open the URL shown in the terminal (usually http://localhost:8501).

You can now ask medical questions and chat with MedBot.

## ⚠️ Limitations

- This project is built for learning and demonstration purposes.
- It only answers questions using the information available in the medical knowledge base.
- If the required information is not available in the documents, the answer may be incomplete or incorrect.
- AI-generated responses should always be verified with trusted medical sources.

## 📌 Disclaimer

This project is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

## 🚀 Future Improvements

- Add source citations for every response
- Support multiple LLMs
- Upload custom PDF documents
- Export chat history
- Voice input and speech output
- Docker deployment
- User authentication

## 👨‍💻 Author

**Omkar Sahu**

Computer Engineering Student | Generative AI & Machine Learning Enthusiast

- LinkedIn: https://www.linkedin.com/in/omkarrsahu/
- GitHub: https://github.com/OmkarSahu0375
