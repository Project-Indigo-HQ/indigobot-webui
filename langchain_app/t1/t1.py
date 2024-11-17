"""
This program uses PyPDFLoader as a file loader and Chroma as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from Gpt4o, Gemini, and Cluade, though more models could be added.
"""

import getpass
import os
import readline
import subprocess

from bs4 import BeautifulSoup
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def load_docs(docs):
    """
    Split text of arg documents into chunks and load them into the Chroma vector store

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


def format_docs(docs) -> str:
    """
    Concatenate chunks to include in prompt

    :param docs: List of documents to format.
    :type docs: list
    :return: Formatted string of document contents.
    :rtype: str
    """
    return "\n\n".join(doc.page_content for doc in docs)


# (Demo/Example code) If Google API key not found, prompt user for it
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API Key:")

llms = [
    ChatOpenAI(model="gpt-4o"),
    GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0),
    ChatAnthropic(model="claude-3-5-sonnet-latest"),
]

model_names = ["GPT-4o", "Gemini 1.5 Pro", "Claude 3.5 Sonnet"]
list_len = len(llms)

# URL list for scraping
urls = [
    "https://github.com/GunterMueller/Books-3/blob/master/Design%20Patterns%20Elements%20of%20Reusable%20Object-Oriented%20Software.pdf",
    "https://rosecityresource.streetroots.org/api/query",
]

# Add local pdf file(s)
file_path = "OWASPtop10forLLMS.pdf"
loader = PyPDFLoader(file_path)
pages = []
for page in loader.lazy_load():
    pages.append(page)

# Add local files
local_path = "."
local_loader = GenericLoader.from_filesystem(
    local_path,
    glob="*",
    # Can select different file suffixes and language types
    suffixes=[".py"],
    parser=LanguageParser(language="python"),
)
local_files = local_loader.load()

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
# No Claude embedding models readily available; instead using Google's
vectorstore.append(
    Chroma(
        persist_directory="./rag_data/.chromadb/gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_query"
        ),
    )
)

load_urls(urls)
load_docs(pages)
load_docs(local_files)

# Create retriever for accessing & displaying doc info & metadata
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

memory = InMemoryChatMessageHistory(session_id="test-session")

instructions = """You are an agent that responds to anything the user asks.
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # First put the history
        ("placeholder", "{chat_history}"),
        # Then the new input
        ("human", "{input}"),
        # Finally the scratchpad
        ("placeholder", "{agent_scratchpad}"),
        instructions,
    ]
)

# TODO
temp = ChatOpenAI(model="gpt-4o")

tools = load_tools(
    ["serpapi", "llm-math", "wikipedia"],
    llm=temp,
    allow_dangerous_tools=False,
)
agent = create_react_agent(temp, tools, prompt)
# NOTE: The verbose setting is great for a look into the model's "thoughts" via stdout
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed;
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}


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
