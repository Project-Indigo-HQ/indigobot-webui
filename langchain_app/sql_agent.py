"""
This program uses PyPDFLoader as a file loader and SQL as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from Gpt4o, Gemini, and Claude, though more models could be added.
"""

import json
import os
import sqlite3

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
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llms = [
    ChatOpenAI(model="gpt-4o"),
    GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0),
    ChatAnthropic(model="claude-3-5-sonnet-latest"),
]

list_len = len(llms)

# init db
DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "rag_data", "indigo_bot_db.sqlite"
)

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# check if the file exists and is a valid db
if not os.path.exists(DB_PATH):
    print("Database file does not exist, creating one...")
    try:
        conn = sqlite3.connect(DB_PATH)
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
    except sqlite3.DatabaseError as e:
        print(f"Error initializing the database: {e}")
        exit()
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents';"
        )
        if not cursor.fetchone():
            cursor.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    metadata TEXT
                );
            """
            )
            conn.commit()
        conn.close()
    except sqlite3.DatabaseError as e:
        print(f"Error opening or reading the database: {e}")
        exit()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# create agents for each llm
agents = []
for llm in llms:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # create llm toolkit
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)  # create agent
    agents.append(agent)


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
        conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
