"""
FastAPI implementation for the IndigoBot RAG system.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from indigobot.__main__ import rag_chain, retriever

# FastAPI app initialization
app = FastAPI(
    title="IndigoBot RAG API",
    description="REST API for RAG-powered question answering about Portland social services",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    input: str
    
    class Config:
        json_schema_extra = {
            "example": {"input": "What housing services are available?"}
        }

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {"answer": "Several housing assistance programs are available..."}
        }

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    response_description="The answer based on retrieved context",
)
async def query_model(request: QueryRequest):
    """Query the RAG pipeline with a question."""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input query cannot be empty")

    try:
        state = {"input": request.input, "chat_history": [], "context": "", "answer": ""}
        response = rag_chain.invoke(state)
        return QueryResponse(answer=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/", summary="Health check", response_description="Basic server status")
async def root():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "IndigoBot RAG API is running!", "version": "1.0.0"}

@app.get(
    "/sources",
    summary="List available sources",
    response_description="List of document sources in the system",
)
async def list_sources():
    """List all document sources available in the vector store."""
    try:
        document_data_sources = set()
        for doc_metadata in retriever.vectorstore.get()["metadatas"]:
            document_data_sources.add(doc_metadata["source"])
        return {"sources": list(document_data_sources)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sources: {str(e)}"
        )

def start_api():
    """Start the FastAPI server"""
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    print(f"\nStarting server on http://{host}:{port}")
    print("To access from another machine, use your VM's external IP address")
    print(f"Make sure your firewall allows incoming traffic on port {port}\n")
    
    try:
        uvicorn.run(
            "indigobot.fastapi:app",
            host=host,
            port=port,
            reload=True,
            access_log=True,
        )
    except Exception as e:
        print(f"Failed to start Uvicorn: {e}")

if __name__ == "__main__":
    try:
        start_api()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
