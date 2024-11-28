"""
This program uses PyPDFLoader as a file loader and SQL as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from Gpt4o, Gemini, and Claude, though more models could be added.
"""

import json
import os
import readline
import sqlite3

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from indigobot.config import GEM_DB, GPT_DB, llms

llm = llms["gpt"]


def init_db():
    """Initialize the SQLite database with required tables"""
    try:
        conn = sqlite3.connect(GPT_DB)
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
        return SQLDatabase.from_uri(f"sqlite:///{GPT_DB}", include_tables=["documents"])
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def load_urls(urls):
    """
    Load documents from URLs into the SQL database

    :param urls: List of URLs to load documents from.
    :type urls: list
    """
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)
        load_docs(splits)


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
        conn = sqlite3.connect(GPT_DB)
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


def query_database(query, params=()):
    """
    Execute a SQL query with optional parameters and return results
    
    :param query: SQL query string
    :param params: Query parameters (optional)
    :return: Query results
    """
    try:
        conn = sqlite3.connect(GPT_DB, timeout=20)
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = cursor.fetchall()
        conn.commit()  # Commit any changes
        return results
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def main():
    # Ensure directory exists
    os.makedirs(os.path.dirname(GPT_DB), exist_ok=True)

    db = init_db()

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # create llm toolkit
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)  # create agent

    retriever = RunnableLambda(
        lambda query=f"SELECT text FROM documents": query_database(query)
    )
    formatted_docs_runnable = RunnableLambda(format_docs)

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = retriever | formatted_docs_runnable | prompt | llm | StrOutputParser()

    while True:
        line = input("llm>> ")
        if line.strip().lower() == "quit":
            print("Exiting the program...")
            break
        if line:
            try:
                result = agent.invoke(line)
                print(f"\n{result}")
            except Exception as e:
                print(f"error: {e}")
        else:
            break


if __name__ == "__main__":
    main()
