"""
This program uses PyPDFLoader as a file loader and SQL as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from Gpt4o, Gemini, and Claude, though more models could be added.
"""

import json
import os
import sqlite3
from pathlib import Path

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

# sql
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if __package__ is None or __package__ == "":
    from config import CURRENT_DIR, R_URLS, RAG_DIR, URLS
else:
    from langchain_app.config import CURRENT_DIR, R_URLS, RAG_DIR, URLS

llms = [ChatAnthropic(model="claude-3-5-sonnet-latest")]

list_len = len(llms)

# init db
GEM_DB = Path(os.path.join(RAG_DIR, ".chromadb/gemini/chroma.sqlite3"))
OAI_DB = Path(os.path.join(RAG_DIR, ".chromadb/openai/chroma.sqlite3"))

# Ensure directory exists
os.makedirs(os.path.dirname(GEM_DB), exist_ok=True)

def init_db():
    """Initialize the SQLite database with required tables"""
    try:
        conn = sqlite3.connect(GEM_DB)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                text TEXT,
                metadata TEXT
            );
            """
        )
        conn.commit()
        conn.close()
        return SQLDatabase.from_uri(
            f"sqlite:///{GEM_DB}",
            include_tables=['documents'],
            # Disable reflection since we know our schema
            reflect_tables=False
        )
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

db = init_db()

# create agents for each llm
agents = []
for llm in llms:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # create llm toolkit
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)  # create agent
    agents.append(agent)


def load_urls(urls):
    """
    Load documents from URLs into the SQL database

    :param urls: List of URLs to load documents from.
    :type urls: list
    """
    docs = AsyncHtmlLoader(urls).load()
    load_docs(docs)


def load_docs(docs):
    """
    Split text of arg documents and load them into the Chroma vector store

    :param docs: List of documents to load and split.
    :type docs: list
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    conn = None
    try:
        conn = sqlite3.connect(GEM_DB)
        cursor = conn.cursor()

        for doc in splits:
            metadata_json = json.dumps(doc.metadata)  # dict to json string
            cursor.execute(
                "INSERT INTO documents (text, metadata) VALUES (?, ?)",
                (doc.page_content, metadata_json),
            )

        conn.commit()
    except sqlite3.DatabaseError as e:
        print(f"Error inserting document into the database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def format_docs(docs):
    """
    Concatenate chunks to include in prompt

    :param docs: List of documents to format.
    :type docs: list
    :return: Formatted string of document contents.
    :rtype: str
    """
    return "\n\n".join(doc.page_content for doc in docs)


def query_database(query):
    conn = sqlite3.connect(GEM_DB)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results


def main():
    retriever = list()
    for i in range(list_len):
        retriever.append(
            RunnableLambda(
                lambda query=f"SELECT text FROM documents": query_database(query)
            )
        )
    formatted_docs_runnable = RunnableLambda(format_docs)

    rag_chain = list()
    prompt = hub.pull("rlm/rag-prompt")

    for i in range(list_len):
        rag_chain.append(
            (
                retriever[i]
                | formatted_docs_runnable
                | prompt
                | llms[i]
                | StrOutputParser()
            )
        )

    print("What kind of questions do you have about the following resources?")
    # iterate over documents and dump metadata
    document_data_sources = set()
    for i in range(list_len):

        for doc_metadata in query_database("SELECT metadata FROM documents"):
            metadata_dict = json.loads(doc_metadata[0])  # JSON to dict
            document_data_sources.add(metadata_dict.get("source", "Unknown"))
    for doc in document_data_sources:
        print(f"  {doc}")

    while True:
        line = input("llm>> ")
        if line.strip().lower() == "quit":
            print("Exiting the program...")
            break
        if line:
            for i, agent in enumerate(agents):
                try:
                    result = agent.invoke(line)
                    print(f"\nModel {i}: {result}")
                except Exception as e:
                    print(f"Model {i} error: {e}")
        else:
            break


if __name__ == "__main__":
    main()
