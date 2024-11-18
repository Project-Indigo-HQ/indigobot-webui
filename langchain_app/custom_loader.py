"""
This module uses PyPDFLoader as a file loader and Chroma as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from either Gpt4o, Gemini, or Claude, though more models could be added.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
from langchain_community.document_transformers import BeautifulSoupTransformer
import unidecode
import re
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from urllib.parse import urljoin
import ssl


def load_docs(docs):
    """
    Split text of arg documents into chunks and load them into the Chroma vector store

    :param docs: List of documents to load and split.
    :type docs: list
    """
    chunks = chunking(docs)
    for i in range(NUM_EMBEDDINGS):
        add_documents(vectorstore[i], chunks, 300)

def load_urls(urls):
    """
    Use AsyncHtmlLoader library to check and scrape websites then load to Chroma

    :param urls: List of URLs to load documents from.
    :type urls: list
    """
    load_docs(AsyncHtmlLoader(urls).load())

def scrape_articles(links):
    """Scrapes list of links, extracts article text, returns Documents"""
    # Scrape list of links
    # Create SSL context with verification disabled
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False  # Disable hostname checking
    ssl_context.verify_mode = ssl.CERT_NONE  # Disable certificate verification
    loader = AsyncHtmlLoader(links, requests_kwargs={"ssl": ssl_context})
    docs = loader.load()
    # Extract article tag
    transformer = BeautifulSoupTransformer()
    docs_tr = transformer.transform_documents(
        documents=docs, tags_to_extract=["article"]
    )
    clean_documents(docs_tr)
    return docs_tr

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

def extract_text(html):
    """Used by loader to extract text from div tag with id of main"""
    soup = BeautifulSoup(html, "html.parser")
    div_main = soup.find("div", {"id": "main"})
    if div_main:
        return div_main.get_text(" ", strip=True)
    return " ".join(soup.stripped_strings)

def clean_documents(documents):
    """Cleans page_content text of Documents list"""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents

def clean_text(text):
    """Replaces unicode characters and strips extra whitespace"""
    text = unidecode.unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_documents(vectorstore, chunks, n):
   """Adds documents to the vectorstore database"""
   for i in range(0, len(chunks), n):
       vectorstore.add_documents(chunks[i:i+n])

def chunking(documents):
    """Takes in Documents and splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    return chunks

def main():
    """
    Execute the document loading process by scraping web pages, reading PDFs, and loading local files.
    """
    load_urls(url_list)
    load_docs(pages)
    load_docs(local_files)

# URL list for scraping
url_list = [
    "https://rosecityresource.streetroots.org/api/query",
]

# Add local pdf file(s)
PDF_PATH = "./rag_data/pdf/LLM_Agents_Beginners.pdf"
loader = PyPDFLoader(PDF_PATH)
pages = []
for page in loader.lazy_load():
    pages.append(page)

# Add local files
LOCALS_PATH = "."
local_loader = GenericLoader.from_filesystem(
    LOCALS_PATH,
    glob="*",
    # Can select different file suffixes and language types
    suffixes=[".py"],
    parser=LanguageParser(language="python"),
)
local_files = local_loader.load()

vectorstore = []
# OpenAI embeddings
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/openai",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
)
# Google embeddings
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_query"
        ),
    )
)
NUM_EMBEDDINGS = len(vectorstore)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
