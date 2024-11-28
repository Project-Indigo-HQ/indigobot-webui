"""
This program uses PyPDFLoader as a file loader and SQL as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from Gpt4o, Gemini, and Claude, though more models could be added.

Usage:
    1. Direct execution:
       python -m indigobot.utils.sql_agent
       
    2. As a module:
       from indigobot.utils.sql_agent import init_db, load_urls, load_docs
       
       # Initialize database
       db = init_db()
       
       # Load URLs into database
       urls = ["https://example.com"]
       load_urls(urls)
       
       # Load documents into database
       from langchain_community.document_loaders import PyPDFLoader
       loader = PyPDFLoader("path/to/doc.pdf")
       docs = loader.load()
       load_docs(docs)
       
    The interactive prompt accepts natural language queries that will be processed
    by the SQL agent to search and analyze the loaded content.
    
    Example queries:
    - "What tables exist in the database?"
    - "Show me the first 5 documents"
    - "Find documents containing the word 'python'"
    
    Type 'quit' to exit the interactive prompt.
"""

import json
import os
import readline
import sqlite3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.utilities import SQLDatabase

from indigobot.config import GEM_DB, GPT_DB, llms

llm = llms["gpt"]


def init_db(db_path=None):
    """Initialize the SQLite database with required tables"""
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
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id INTEGER,
                key TEXT NOT NULL,
                string_value TEXT,
                int_value INTEGER,
                float_value REAL,
                bool_value INTEGER,
                PRIMARY KEY (id, key),
                FOREIGN KEY(id) REFERENCES embeddings (id)
            );
        """
        )
        conn.commit()
        conn.close()

        # Return SQLDatabase instance
        return SQLDatabase.from_uri(
            f"sqlite:///{db_file}", include_tables=["embedding_metadata"]
        )
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def load_urls(urls, db_path=None):
    """
    Load documents from URLs into the SQL database
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
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    conn = None
    try:
        db_file = db_path or GPT_DB
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        for doc in splits:
            # First insert into embeddings table to get an id
            cursor.execute("INSERT INTO embeddings DEFAULT VALUES")
            doc_id = cursor.lastrowid
            
            # Insert the document content
            cursor.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                (doc_id, "content", doc.page_content)
            )
            
            # Insert each metadata field
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    cursor.execute(
                        "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                        (doc_id, key, value)
                    )
                elif isinstance(value, bool):
                    cursor.execute(
                        "INSERT INTO embedding_metadata (id, key, bool_value) VALUES (?, ?, ?)",
                        (doc_id, key, int(value))
                    )
                elif isinstance(value, int):
                    cursor.execute(
                        "INSERT INTO embedding_metadata (id, key, int_value) VALUES (?, ?, ?)",
                        (doc_id, key, value)
                    )
                elif isinstance(value, float):
                    cursor.execute(
                        "INSERT INTO embedding_metadata (id, key, float_value) VALUES (?, ?, ?)",
                        (doc_id, key, value)
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
    db = init_db(GPT_DB)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # create llm toolkit
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Limit retry attempts
        early_stopping_method="generate"  # Use generate method for early stopping
    )  # create agent

    # retriever = RunnableLambda(
    #     lambda query=f"SELECT text FROM embedding_metadata": query_database(query)
    # )
    # formatted_docs_runnable = RunnableLambda(format_docs)

    # prompt = hub.pull("rlm/rag-prompt")

    # rag_chain = retriever | formatted_docs_runnable | prompt | llm | StrOutputParser()

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
