"""
This is meant to be a starting point for the Indigo-CfSS model
.............................................................
"""

import readline  # need this to use arrow keys
from typing import Sequence
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


import custom_loader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

if __package__ is None or __package__ == "":
    # Use current directory visibility
    import custom_loader
else:
    # Use package visibility
    from langchain_app import custom_loader


# Define API models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    input: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": "What are the key concepts of LLM agents?"
            }
        }

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    context: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "LLM agents are AI systems that can...",
                "context": "Retrieved from documentation..."
            }
        }

class ChatState(TypedDict):
    """
    A dictionary that represents the state of a chat interaction/history.
    """
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

class State(BaseModel):
    """
    Pydantic model for chat state validation
    """
    input: str
    chat_history: list[dict] = []
    context: str = ""
    answer: str = ""


def call_model(state: State):
    """
    Call the model with the given state and return the response.

    :param state: The state containing input, chat history, context, and answer.
    :type state: State
    :return: Updated state with chat history, context, and answer.
    :rtype: dict
    """
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


# currently not used
def search_vectorstore(query):
    """
    Perform a similarity search in the vector store with the given query.

    :param query: The query to search for.
    :type query: str
    """
    docs = vectorstore.similarity_search(query)
    print(f"Query database for: {query}")
    if docs:
        print(f"Closest document match in database: {docs[0].metadata['source']}")
    else:
        print("No matching documents")


# Model to use
llm = ChatOpenAI(model="gpt-4o")
# llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# OpenAI embeddings
vectorstore = Chroma(
    persist_directory="./rag_data/.chromadb/openai",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
)
GPT_SQL_DB = "./rag_data/.chromadb/openai/chroma.sqlite3"

# Google embeddings
# vectorstore = Chroma(
#     persist_directory="./rag_data/.chromadb/gemini",
#     embedding_function=GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001", task_type="retrieval_query"
#     ),
# )
# GOOGLE_SQL_DB = "./rag_data/.chromadb/gemini/chroma.sqlite3"

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

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
'''
        '''
# FastAPI app initialization
app = FastAPI(
    title="RAG API",
    description="REST API for RAG-powered question answering",
    version="1.0.0"
)

# Define API endpoints
@app.post("/query", response_model=QueryResponse, 
          summary="Query the RAG system",
          response_description="The answer and supporting context")
async def query_model(request: QueryRequest):
    """
    Query the RAG pipeline with a question.
    
    The system will:
    1. Retrieve relevant context from the document store
    2. Generate an answer based on the context
    3. Return both the answer and the supporting context
    
    Raises:
        HTTPException(400): If the input is invalid
        HTTPException(500): If there's an internal error
    """
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input query cannot be empty")
        
    try:
        response = rag_chain.invoke({"input": request.input})
        return QueryResponse(
            answer=response["answer"],
            context=response.get("context", "No context available")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/", 
         summary="Health check",
         response_description="Basic server status")
async def root():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "message": "RAG API is running!",
        "version": "1.0.0"
    }

@app.get("/sources",
         summary="List available sources",
         response_description="List of document sources in the system")
async def list_sources():
    """
    List all document sources available in the vector store.
    """
    try:
        document_data_sources = set()
        for doc_metadata in retriever.vectorstore.get()["metadatas"]:
            document_data_sources.add(doc_metadata["source"])
        return {"sources": list(document_data_sources)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sources: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    # Optionally, execute custom loader before starting the server
    load_res = input("Would you like to execute the loader? (y/n) ")
    if load_res == "y":
        custom_loader.main()

    # Start LangServe
    print("Starting LangServe API at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
