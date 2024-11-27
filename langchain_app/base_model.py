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

import sys
from pathlib import Path
file_path = Path(__file__).resolve()
parent_dir = file_path.parent.parent
sys.path.append(str(parent_dir))


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
    chat_history: Annotated[Sequence[BaseMessage], add_messages] = []
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
llm = ChatOpenAI(model="gpt-4")
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

# Create vectorstore retriever with limits to avoid context overflow
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,  # Limit to top 3 most relevant documents
        "fetch_k": 4,  # Fetch 4 documents before selecting top 3
    }
)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Reformulate the user's question into a standalone question, "
    "considering the chat history. Return the original question if no reformulation needed."
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
    "Answer questions using the provided context. "
    "Keep answers concise, max 3 sentences. "
    "Say 'I don't know' if unsure.\n\n{context}"
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
        # Initialize state with empty chat history if none provided
        state = {
            "input": request.input,
            "chat_history": [],
            "context": ""
        }
        response = rag_chain.invoke(state)
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
    import os
    import sys
    
    
    # Optionally, execute custom loader before starting the server
    load_res = input("Would you like to execute the loader? (y/n) ")
    if load_res == "y":
        custom_loader.main()

    # Start FastAPI
    print("Starting FastAPI server at http://localhost:8000")
    uvicorn.run("langchain_app.base_model:app", host="localhost", port=8000, reload=True)
