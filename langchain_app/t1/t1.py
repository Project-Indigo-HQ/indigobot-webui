"""
This program uses PyPDFLoader as a file loader and Chroma as a vector database.
It loads local PDFs, Python files, and also checks web pages to scrape and consume data.
It currently gets responses from either Gpt4o, Gemini, or Claude, though more models could be added.
"""

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmCategory,
    HarmBlockThreshold,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated, TypedDict
from typing import Sequence
import readline  # need this to use arrow keys


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def load_docs(docs):
    """
    Split text of arg documents into chunks and load them into the Chroma vector store

    :param docs: List of documents to load and split.
    :type docs: list
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(documents=splits)


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


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


def search_vectorstore(query):
    docs = vectorstore.similarity_search(query)
    print(f"Query database for: {query}")
    if docs:
        print(f"Closest document match in database: {docs[0].metadata['source']}")
    else:
        print("No matching documents")


llm = ChatOpenAI(model="gpt-4o")
# llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# URL list for scraping
urls = [
    "https://rosecityresource.streetroots.org/api/query",
]

# Add local pdf file(s)
file_path = "./rag_data/pdf/LLM_Agents_Beginners.pdf"
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

# OpenAI embeddings
vectorstore = Chroma(
    persist_directory="./rag_data/.chromadb/openai",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
)
gpt_db = "./rag_data/.chromadb/openai/chroma.sqlite3"

# Google embeddings
# vectorstore = Chroma(
#     persist_directory="./rag_data/.chromadb/gemini",
#     embedding_function=GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001", task_type="retrieval_query"
#     ),
# )
# google_db = "./rag_data/.chromadb/gemini/chroma.sqlite3"

# Create vectorstore retriever for accessing & displaying doc info & metadata
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# db = SQLDatabase.from_uri(f"sqlite:///{gpt_db}")
# sql_tool = SQLDatabaseToolkit(db=db, llm=llm)
# agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

retriever_tool = create_retriever_tool(retriever, "my_retriever", "my_description")
tools = [retriever_tool]

config = {"configurable": {"thread_id": "abc123"}}

load_urls(urls)
load_docs(pages)
load_docs(local_files)

print("What kind of questions do you have about the following resources?")
# Iterate over documents and dump metadata
document_data_sources = set()
for doc_metadata in retriever.vectorstore.get()["metadatas"]:
    document_data_sources.add(doc_metadata["source"])
for doc in document_data_sources:
    print(f"  {doc}")

while True:
    try:
        line = input("llm>> ")
        if line:
            result = app.invoke(
                {"input": line},
                config=config,
            )
            print(result["answer"])
        else:
            break
    except Exception as e:
        print(e)
