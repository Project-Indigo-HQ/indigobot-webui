import json
import re
import os
import requests
import unidecode
import itertools
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
import urllib3

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def clean_text(text):
    """Replaces unicode characters and strips extra whitespace"""
    text = unidecode.unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_documents(documents):
    """Cleans page_content text of Documents list"""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents

def scrape_articles(links):
    """Scrapes list of links, extracts article text, returns Documents"""
    # Scrape list of links
    loader = AsyncHtmlLoader(links, requests_kwargs={"verify":False})
    docs = loader.load()
    # Extract article tag
    transformer = BeautifulSoupTransformer()
    docs_tr = transformer.transform_documents(
        documents=docs, tags_to_extract=["article"]
    )
    clean_documents(docs_tr)
    return docs_tr

def load_pdf_documents(dir):
    """Loads all PDFs in given directory"""
    loader = PyPDFDirectoryLoader(dir)
    docs = loader.load()
    return docs

def chunking(documents):
    """Takes in Documents and splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    return chunks

def extract_text(html):
    """Used by loader to extract text from div tag with id of main"""
    soup = BeautifulSoup(html, "html.parser")
    div_main = soup.find("div", {"id": "main"})
    if div_main:
        return div_main.get_text(" ", strip=True)
    return " ".join(soup.stripped_strings)

def scrape_main(url, depth):
    """Recursively scrapes URL and returns Documents"""
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=depth,
        timeout=20,
        use_async=True,
        prevent_outside=True,
        check_response_status=True,
        continue_on_failure=True,
        extractor=extract_text,
    )
    docs = loader.load()
    clean_documents(docs)
    return docs

def load_config(filename):
    """Reads configuration from a JSON file"""
    with open(filename, "r") as f:
        config = json.load(f)
    return config

def add_documents(vectorstore, chunks, n):
   for i in range(0, len(chunks), n):
       print(f"{i} of {len(chunks)}")
       vectorstore.add_documents(chunks[i:i+n])

if __name__ == "__main__":
    # loading environment variables
    load_dotenv()

    # Check if OpenAI API key is provided in environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        # Prompt the user toenter the OpenAI API key if not found
        openai_api_key = input("Please enter your OpenAI API key: ")
        # Set the environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    # Load configuration
    config = load_config("config.json")
    bulletin_websites = config["bulletin_websites"]
    cs_website = config["cs_website"]
    cs_courses_websites = config["cs_courses_websites"]
    urllib3.disable_warnings()

    # Initialize vectorstore
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./.chromadb"
    )

    # Gets all the relevent URLs from the CS department landing page, 
    # scrapes them, chunks them, then adds them to vector database
    resp = requests.get(cs_website)
    soup = BeautifulSoup(resp.text,"html.parser")
    links = list({urljoin(cs_website,a['href']) for a in soup.find_all('a', href=True) if any(['computer-science' in a['href'], 'security' in a['href']])})
    documents = scrape_articles(links)
    chunks = chunking(documents)
    add_documents(vectorstore, chunks, 300)

    # Gets all the relevent URLs from the undergraduate and graduate course pages.  (Leave out for now due to long run-time)

    # for website in cs_courses_websites:
    #     start_time = time.time()
    #     resp = requests.get(website)
    #     soup = BeautifulSoup(resp.text,"html.parser")
    #     links = list({a['href'] for a in soup.find_all('a', href=True) if 'docs.google.com/document' in a['href']})
    #     loader = AsyncHtmlLoader(links, requests_kwargs={"verify":False})
    #     docs = loader.load()
    #     chunks = chunking(docs)
    #     add_documents(vectorstore, chunks, 300) # Create embeddings and save them in a vector store
    #     elapsed_time = time.time() - start_time
    #     print(f"time elapsed: {elapsed_time}")

    # Loads all PDF documents in FAQ directory into vector database
    docs = load_pdf_documents("FAQ")  # Load all documents in the directory(success)
    chunks = chunking(docs)  # Split documents into chunks
    add_documents(vectorstore, chunks, 300) # Create embeddings and save them in a vector store

    # Scrapes URLs given for Academic Bulletin recursively, chunks them,
    # then adds them to vector database
    for website in bulletin_websites:
        docs = scrape_main(website, 12)
        chunks = chunking(docs)
        add_documents(vectorstore, chunks, 300) # Create embeddings and save them in a vector store

    retriever = vectorstore.as_retriever()
    docs = retriever.vectorstore.get()
    print(docs['metadatas'])
    print(docs['documents'])
