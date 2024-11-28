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


def init_db(db_path=None):
    """Initialize the SQLite database with required tables

    Args:
        db_path (str, optional): Override default database path. Defaults to None.
    """
    try:
        # Use provided path or default
        db_file = db_path or GPT_DB

        # Ensure the database directory exists
        os.makedirs(os.path.dirname(db_file), exist_ok=True)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create the documents table if it doesn't exist
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

        # Return SQLDatabase instance
        return SQLDatabase.from_uri(
            f"sqlite:///{db_file}", include_tables=["documents"]
        )
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def load_urls(urls, db_path=None):
    """
    Load documents from URLs into the SQL database

    Args:
        urls (list): List of URLs to load documents from
        db_path (str, optional): Override default database path. Defaults to None.
    """
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=10
        )
        splits = text_splitter.split_documents(docs)
        load_docs(splits, db_path=db_path)


def load_docs(docs, db_path=None):
    """
    Split text of arg documents and load them into the SQL database

    Args:
        docs (list): List of documents to load and split
        db_path (str, optional): Override default database path. Defaults to None.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    conn = None
    try:
        db_file = db_path or GPT_DB
        conn = sqlite3.connect(db_file)
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


def query_database(query, params=(), db_path=None):
    """
    Execute a SQL query with optional parameters and return results

    Args:
        query (str): SQL query string
        params (tuple, optional): Query parameters. Defaults to ().
        db_path (str, optional): Override default database path. Defaults to None.

    Returns:
        list: Query results
    """
    try:
        db_file = db_path or GPT_DB
        conn = sqlite3.connect(db_file, timeout=20)
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
        if "conn" in locals():
            conn.rollback()
        raise
    finally:
        if "conn" in locals():
            conn.close()


def main():
    # Ensure directory exists
    os.makedirs(os.path.dirname(GPT_DB), exist_ok=True)

    db = init_db(GPT_DB)

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
