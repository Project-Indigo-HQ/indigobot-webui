"""
This is the main chatbot program for conversational capabilities and info distribution.
"""

import os
import readline  # Required for using arrow keys in CLI
import sys
from pathlib import Path
from typing import Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from indigobot.config import llms, vectorstores
from indigobot.utils import custom_loader

llm = llms["gpt"]

# file_path = Path(__file__).resolve()
# parent_dir = file_path.parent.parent
# sys.path.append(str(parent_dir))


# Define API models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""

    input: str

    class Config:
        json_schema_extra = {
            "example": {"input": "What are the key concepts of LLM agents?"}
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""

    answer: str

    class Config:
        json_schema_extra = {
            "example": {"answer": "LLM agents are AI systems that can..."}
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


# FastAPI app initialization
app = FastAPI(
    title="RAG API",
    description="REST API for RAG-powered question answering",
    version="1.0.0",
)


# Define API endpoints
@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    response_description="The answer and supporting context",
)
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
        state = State(input=request.input, chat_history=[], context="").dict()
        response = rag_chain.invoke(state)
        # Format context from documents into a concise string
        context = ""
        if isinstance(response.get("context"), list):
            # Extract just the service descriptions from the documents
            contexts = []
            for doc in response["context"]:
                content = doc.page_content
                # Try to extract service description if it exists
                if "service_description" in content:
                    try:
                        import json

                        data = json.loads("{" + content.split("{", 1)[1])
                        desc = data.get("service_description", "")
                        if desc:
                            if len(desc) > 150:
                                desc = desc[:150] + "..."
                            contexts.append(desc)
                    except:
                        # Fallback to simple truncation if JSON parsing fails
                        if len(content) > 150:
                            content = content[:150] + "..."
                        contexts.append(content)
                else:
                    # Simple truncation for non-service content
                    if len(content) > 150:
                        content = content[:150] + "..."
                    contexts.append(content)

            context = "\n• ".join(contexts)
        else:
            context = str(response.get("context", "No context available"))
            if len(context) > 450:
                context = context[:450] + "..."

        return QueryResponse(answer=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/", summary="Health check", response_description="Basic server status")
async def root():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "message": "RAG API is running!", "version": "1.0.0"}


@app.get(
    "/sources",
    summary="List available sources",
    response_description="List of document sources in the system",
)
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
            status_code=500, detail=f"Error retrieving sources: {str(e)}"
        )


retriever = vectorstores["gpt"].as_retriever()

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
    "You are an assistant that answers questions/provides information about "
    "social services in Portland, Oregon. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, "
    "say that you don't know. Use three sentences maximum and keep the answer concise."
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

retriever_tool = create_retriever_tool(retriever, "my_retriever", "my_description")
tools = [retriever_tool]

# Configuration constants
thread_config = {"configurable": {"thread_id": "abc123"}}


def main(skip_loader: bool = False) -> None:
    """
    Main function that runs the interactive chat loop.
    Handles user input and displays model responses.
    Exits when user enters an empty line.

    Args:
        skip_loader (bool): If True, skips the loader prompt. Useful for testing.
    """
    if not skip_loader:
        load_res = input("Would you like to execute the loader? (y/n) ")
        if load_res == "y":
            custom_loader.main()

    # Start FastAPI
    # Get port from environment variable or use default 8000
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Explicitly bind to all interfaces
    print(f"\nStarting server on http://{host}:{port}")
    print("To access from another machine, use your VM's external IP address")
    print(f"Make sure your GCP firewall allows incoming traffic on port {port}\n")

    try:
        uvicorn.run(
            "indigobot.__main__:app",
            host=host,
            port=port,
            reload=True,
            access_log=True,  # Enable access logging
        )
    except Exception as e:
        print(f"Failure running Uvicorn: {e}")

    while True:
        try:
            print()
            line = input("llm>> ")
            if line:
                result = app.invoke(
                    {"input": line},
                    config=thread_config,
                )
                print()
                print(result["answer"])
            else:
                break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        main(skip_loader=False)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
