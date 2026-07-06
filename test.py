print("Start")

from langchain_huggingface import HuggingFaceEmbeddings

print("Imported")

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Done")