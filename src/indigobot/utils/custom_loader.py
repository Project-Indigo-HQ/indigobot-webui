"""
A customized document loader for processing and storing various document types.
The module uses Chroma as a vector database for storing processed documents. It includes
utilities for text cleaning, chunking, and batch processing of documents.

Functions:
    clean_text: Cleans and normalizes text content
    clean_documents: Processes a list of documents
    chunking: Splits documents into manageable chunks
    load_docs: Loads documents into the vector store
    load_urls: Processes URLs and loads their content
    scrape_urls: Scrapes and processes website content
    start_loader: Main entry point for document loading process
"""

import os
import re

import unidecode
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from indigobot.config import RAG_DIR, r_url_list, url_list, vectorstore, cls_url_list
from indigobot.utils.jf_crawler import crawl
from indigobot.utils.refine_html import load_JSON_files, refine_text


def clean_text(text):
    """
    Replaces unicode characters and strips extra whitespace from text.

    :param text: Raw text content to be cleaned
    :type text: str
    :return: Text with unicode characters replaced and whitespace normalized
    :rtype: str
    :raises UnicodeError: If unicode replacement fails
    """
    text = unidecode.unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_documents(documents):
    """
    Cleans the page_content text of a list of Documents.

    Processes each document in the list by cleaning its page_content
    using the clean_text() function.

    :param documents: List of Document objects to clean
    :type documents: list[Document]
    :return: List of Document objects with cleaned page_content
    :rtype: list[Document]
    :raises AttributeError: If documents don't have page_content attribute
    """
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents


def chunking(documents):
    """
    Splits text of documents into smaller chunks for processing.

    Uses RecursiveCharacterTextSplitter to break documents into chunks
    of approximately 10000 characters with 1000 character overlap.

    :param documents: List of Document objects to split
    :type documents: list[Document]
    :return: List of Document chunks
    :rtype: list[Document]
    :raises ValueError: If documents cannot be split properly
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_docs(docs):
    """
    Split text of documents into chunks and load them into the Chroma vector store.

    Processes documents by:
    1. Splitting them into chunks using chunking()
    2. Adding chunks to the vector store in batches of 300

    :param docs: List of Document objects to process and load
    :type docs: list[Document]
    :raises Exception: If chunking operations fail
    """

    chunks = chunking(docs)
    add_docs(chunks, 300)


def load_urls(urls):
    """
    Asynchronously load and process web pages from URLs into the vector store.

    Uses AsyncHtmlLoader to fetch web pages concurrently, then processes
    and loads them into the Chroma vector store.

    :param urls: List of URLs to scrape and process
    :type urls: list[str]
    :raises Exception: If URL loading or processing fails
    """
    load_docs(AsyncHtmlLoader(urls).load())


def extract_text(html):
    """
    Extracts text from a div tag with id of 'main' from HTML content.

    Attempts to find and extract text from a div with id='main',
    falling back to all text content if the main div isn't found.

    :param html: Raw HTML content to parse
    :type html: str
    :return: Extracted text content with normalized spacing
    :rtype: str
    :raises BeautifulSoupError: If HTML parsing fails
    """
    soup = BeautifulSoup(html, "html.parser")
    div_main = soup.find("div", {"id": "main"})
    if div_main:
        return div_main.get_text(" ", strip=True)
    return " ".join(soup.stripped_strings)


def scrape_main(url, depth):
    """
    Recursively scrapes a URL and its linked pages up to specified depth.

    Uses RecursiveUrlLoader with async loading and safety checks:
    - Prevents scraping outside the original domain
    - Checks response status codes
    - Uses timeouts to prevent hanging
    - Continues on individual page failures

    :param url: The base URL to start scraping from
    :type url: str
    :param depth: Maximum recursion depth for following links
    :type depth: int
    :return: List of Document objects with cleaned content
    :rtype: list[Document]
    :raises Exception: If scraping fails or timeout occurs
    """
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


def add_docs(chunks, n):
    """
    Adds document chunks to the vector store in batches.

    Processes chunks in batches of size n to prevent memory issues
    and optimize vector store operations.

    :param chunks: List of Document chunks to add
    :type chunks: list[Document]
    :param n: Batch size for adding documents
    :type n: int
    :raises Exception: If vector store operations fail
    """
    for i in range(0, len(chunks), n):
        vectorstore.add_documents(chunks[i : i + n])


def scrape_urls(urls):
    """
    Processes multiple URLs by scraping and loading them into the vector store.

    For each URL:
    1. Scrapes content recursively using scrape_main()
    2. Splits content into chunks
    3. Adds chunks to vector store in batches

    :param urls: List of URLs to process
    :type urls: list[str]
    :raises Exception: If scraping or processing fails for any URL
    """
    for url in urls:
        docs = scrape_main(url, 12)
        chunks = chunking(docs)
        add_docs(chunks, 300)


def jf_loader():
    """
    Fetches and refines documents from the website source and loads them into the vector database.

    Process flow:
    1. Crawls website using crawl()
    2. Refines text content using refine_text()
    3. Loads processed JSON files from the crawl_temp directory
    4. Adds documents to the vector store

    :raises Exception: If crawling, refinement, or loading fails
    """

    # Fetching document from website then save to for further process
    crawl()

    # # Refine text by removing meanless conent from the XML files
    refine_text()

    # Load the content into vectorstore database
    JSON_DOCS_DIR = os.path.join(RAG_DIR, "crawl_temp/processed_text")
    json_docs = load_JSON_files(JSON_DOCS_DIR)
    print(f"Loaded {len(json_docs)} documents.")

    load_docs(json_docs)


def start_loader():
    """
    Execute the document loading process by scraping web pages and loading local files.

    Main entry point for document processing that:
    1. Iterates through configured vector stores
    2. Processes URLs from r_url_list and url_list
    3. Loads documents using jf_loader
    4. Handles errors for each vector store independently

    :raises Exception: If loading fails for all vector stores
    """
    try:
        scrape_urls(r_url_list)
        scrape_urls(cls_url_list)
        load_urls(url_list)
        jf_loader()
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise


if __name__ == "__main__":
    try:
        start_loader()
    except Exception as e:
        print(e)
