import streamlit as st
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺"
)

st.title("🩺 Medical Chatbot")
st.write("Ask your medical question below.")

# Load embeddings
embeddings = download_embeddings()

# Connect to Pinecone
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize LLM
chatmodel = ChatGroq(
    model_name="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create RAG Chain
question_answering_chain = create_stuff_documents_chain(
    chatmodel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answering_chain
)

# Chat UI
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Type your medical question...")

if question:
    # Store and display user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    # Generate response
    with st.spinner("Generating response..."):
        response = rag_chain.invoke({"input": question})
        answer = response["answer"]

    # Store and display assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)