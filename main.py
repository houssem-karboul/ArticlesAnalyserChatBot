import os
import streamlit as st
import pickle
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain.llms.openllm import OpenLLM
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Use environment variables instead of hardcoded values
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_embeddings_api_key = os.getenv("NVIDIA_EMBEDDINGS_API_KEY")


st.title("Articles Analyser ChatBot 📈")
st.sidebar.title("Articles URLs")

# store websites URLS

urls = []

# Articles websites Urls.

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# initilize llm model
main_placeholder = st.empty()



llm=ChatNVIDIA(api_key=nvidia_api_key,
               model="mistralai/mixtral-8x22b-instruct-v0.1") 

if  process_url_clicked:

    # load data
    #loader = UnstructuredURLLoader(urls=urls, mode="elements",)
    loader = UnstructuredURLLoader(urls=urls, mode="elements", metadata_fn=lambda url: {"source": url})
    data = loader.load()
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=3000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    # docs = text_splitter.split_documents(data)
    
    docs = text_splitter.split_documents(data)

    for doc in docs:
        if 'source' not in doc.metadata:
            doc.metadata['source'] = doc.metadata['url']


    # Create embeddings

    embeddings = NVIDIAEmbeddings(
        nvidia_api_key=nvidia_embeddings_api_key,
        model= "nvidia/nv-embedqa-e5-v5" # Specify the model explicitly
    )


    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    
    for doc in docs:
        if 'source' not in doc.metadata:
            doc.metadata['source'] = 'Unknown'

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)



query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)