print("App is starting...")

print("Importing Flask...")
from flask import Flask, jsonify, render_template, request
print("✓ Flask imported")

print("Importing helper...")
from src.helper import download_embeddings
print("✓ helper imported")

print("Importing Pinecone...")
from langchain_pinecone import PineconeVectorStore
print("✓ Pinecone imported")

print("Importing Groq...")
from langchain_groq import ChatGroq
print("✓ Groq imported")

print("Importing prompts...")
from langchain_core.prompts import ChatPromptTemplate
print("✓ ChatPromptTemplate imported")

print("Importing chains...")
from langchain_classic.chains import create_retrieval_chain
print("✓ Retrieval chain imported")

print("Importing combine_documents...")
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
print("✓ Stuff documents chain imported")

print("Importing dotenv...")
from dotenv import load_dotenv
print("✓ dotenv imported")

print("Importing prompt...")
from src.prompt import *
print("✓ prompt imported")

import os
print("✓ os imported")

print("App Started.....")
app = Flask(__name__)
print("1. Flask app created")

load_dotenv()
print("2. Environment loaded")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

embeddings = download_embeddings()
print("3. Embeddings loaded")

index_name = "medical-chatbot"
# embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
print("4. Pinecone connected")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={'k':3})
print("5. Retriever created")

chatmodel = ChatGroq(model_name = "openai/gpt-oss-20b", api_key=GROQ_API_KEY)
print("6. LLM initialized")

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', "{input}")
    ]
)

question_answering_chain =  create_stuff_documents_chain(chatmodel, prompt)
print("7. QA chain ready")

rag_chain = create_retrieval_chain(retriever, question_answering_chain)
print("8. RAG chain ready")

@app.route("/")
def index():
    return render_template('chat.html')
print(app.url_map)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({'input': msg})
    print("Response: ", response['answer'])
    return str(response['answer'])



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)