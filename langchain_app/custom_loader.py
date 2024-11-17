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


def load_docs(docs):
    """
    Split text of arg documents into chunks and load them into the Chroma vector store

    :param docs: List of documents to load and split.
    :type docs: list
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    for i in range(NUM_EMBEDDINGS):
        vectorstore[i].add_documents(documents=splits)


def load_urls(urls):
    """
    Use AsyncHtmlLoader library to check and scrape websites then load to Chroma

    :param urls: List of URLs to load documents from.
    :type urls: list
    """
    load_docs(AsyncHtmlLoader(urls).load())


def main():
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
