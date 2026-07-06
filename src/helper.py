from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings



# Extract data from the pdf files
def load_pdf_files(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# Filtering the metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document object, return a new list of Document object containing only 'source' in the metadata and hte original page_content.
    """
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get('source')
        page = doc.metadata.get('page')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={'source':src, 'page': page}
            )
        )

    return minimal_docs

# split the data into chunks
def split_text(minimal_docs: List[Document], chunk_size = 1000, chunk_overlap = 200, length_function = len) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(minimal_docs)

# dowloading the embedding model from huggingface
def download_embeddings():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model
    )
    return embeddings