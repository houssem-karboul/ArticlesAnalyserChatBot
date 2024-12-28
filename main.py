import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Constants
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBEDDINGS_API_KEY = os.getenv("NVIDIA_EMBEDDINGS_API_KEY")
FAISS_STORE_PATH = "faiss_store_openai.pkl"
MAX_URLS = 3

# Streamlit UI setup
st.title("Articles Analyser ChatBot 📈")
st.sidebar.title("Articles URLs")

# Collect URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(MAX_URLS)]
process_url_clicked = st.sidebar.button("Process URLs")

# Initialize main placeholder
main_placeholder = st.empty()

# Initialize LLM model
llm = ChatNVIDIA(api_key=NVIDIA_API_KEY, model="mistralai/mixtral-8x22b-instruct-v0.1")

def process_urls(urls):
    # Load and process data
    loader = UnstructuredURLLoader(urls=urls, mode="elements", metadata_fn=lambda url: {"source": url})
    data = loader.load()
    main_placeholder.text("Data Loading...Started...✅✅✅")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=3000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Ensure source metadata
    for doc in docs:
        if 'source' not in doc.metadata:
            doc.metadata['source'] = doc.metadata.get('url', 'Unknown')

    # Create embeddings
    embeddings = NVIDIAEmbeddings(nvidia_api_key=NVIDIA_EMBEDDINGS_API_KEY, model="nvidia/nv-embedqa-e5-v5")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    # Save FAISS index
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump(vectorstore_openai, f)

def answer_query(query):
    if os.path.exists(FAISS_STORE_PATH):
        with open(FAISS_STORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

# Main execution
if process_url_clicked:
    process_urls(urls)

query = main_placeholder.text_input("Question: ")
if query:
    answer_query(query)
