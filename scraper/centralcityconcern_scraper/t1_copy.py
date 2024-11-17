"""
This program uses Chroma as a vector database to store streetroots JSON blob.
It scrapes the JSON blob using Beautifulsoup
It currently gets responses from Gpt4o, Gemini, and Cluade, though more models could be added.
"""

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, AsyncHtmlLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
import getpass
import os
import json

llms = [
    ChatOpenAI(model="gpt-4o"),
    GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0),
    ChatAnthropic(model="claude-3-5-sonnet-latest"),
]

model_names = ["GPT-4", "Gemini 1.5 Pro", "Claude 3.5 Sonnet"]
list_len = len(llms)

def load_docs(docs):
    """
    Split text of arg documents and load them into the Chroma vector store

    :param docs: List of documents to load and split.
    :type docs: list
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    for i in range(list_len):
        vectorstore[i].add_documents(documents=splits)

def load_urls(urls):
    """
    Use AsyncHtmlLoader library to check and scrape websites then load to Chroma

    :param urls: List of URLs to load documents from.
    :type urls: list
    """
    load_docs(AsyncHtmlLoader(urls).load())

def load_JSON_files(folder_path):
    JSON_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_content = json.load(file)
                        if isinstance(json_content, dict) and "content" in json_content:
                            JSON_files.append(Document(
                                page_content=json_content["content"],
                                metadata=json_content.get("metadata", {})
                            ))
                        elif isinstance(json_content, list):
                            for item in json_content:
                                if isinstance(item, dict) and "content" in item:
                                    JSON_files.append(Document(
                                        page_content=item["content"],
                                        metadata=item.get("metadata", {})
                                    ))
            except Exception as e:
                print(f"Error reading JSON file {file_path}: {e}")

    return JSON_files

def format_docs(docs):
    """
    Concatenate chunks to include in prompt

    :param docs: List of documents to format.
    :type docs: list
    :return: Formatted string of document contents.
    :rtype: str
    """
    return "\n\n".join(doc.page_content for doc in docs)

# URL list for scraping
urls = [
    "https://rosecityresource.streetroots.org/api/query",
]

# (Demo/Example code) If Google API key not found, prompt user for it
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API Key:")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI API Key:")
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("ANTHROPIC API Key:")

# Create a list of vectorstore entries for each model embedding
vectorstore = list()
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/openai",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
)
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_query"
        ),
    )
)
# No Claude embedding models readily available; just using Google's
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_query"
        ),
    )
)

# Load JSON documents into the vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
json_files_dir = os.path.join(script_dir, "processed_text")
JSON_files = load_JSON_files(json_files_dir)
for file in JSON_files:
    load_docs(file)

retriever = list()
for i in range(list_len):
    retriever.append(vectorstore[i].as_retriever())

rag_chain = list()
prompt = hub.pull("rlm/rag-prompt")
for i in range(list_len):
    rag_chain.append(
        (
            {"context": retriever[i] | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llms[i]
            | StrOutputParser()
        )
    )

print("What kind of questions do you have about the following resources?")
# Iterate over documents and dump metadata
document_data_sources = set()
for i in range(list_len):
    for doc_metadata in retriever[i].vectorstore.get()["metadatas"]:
        document_data_sources.add(doc_metadata["source"])
for doc in document_data_sources:
    print(f"  {doc}")

while True:
    line = input("llm>> ")
    if line:
        for i in range(list_len):
            result = rag_chain[i].invoke(line)
            print(f"\nModel {model_names[i]}: {result}")
    else:
        break
