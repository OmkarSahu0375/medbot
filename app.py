from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.helper import download_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)
rag_chain = None

def initialize_rag():
    global rag_chain

    print("Initializing embeddings...")

    # load only when needed (NOT at startup)
    embeddings = download_embeddings()

    print("Connecting to Pinecone...")

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )

    print("Loading LLM...")

    chatmodel = ChatGroq(
        model_name="openai/gpt-oss-20b",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{input}")
    ])

    qa_chain = create_stuff_documents_chain(chatmodel, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    print("RAG initialized successfully!")
    return rag_chain

# ROUTES
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    global rag_chain

    # initialize ONLY on first request
    if rag_chain is None:
        rag_chain = initialize_rag()

    msg = request.form["msg"]

    response = rag_chain.invoke({"input": msg})

    return str(response["answer"])



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)