"""
FastAPI application providing a REST API for the Chinook music database.

This module implements CRUD operations for artists, genres, albums, and tracks.
It uses SQLAlchemy for database operations and Pydantic for data validation.
"""

import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from indigobot.config import RAG_DIR
from .init_db import (
    Base,
    SessionLocal,
    engine,
    init_sql_db,
)

# Ensure database directory exists
os.makedirs(RAG_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Initialize database with sample data
init_sql_db()


# Create database tables
Base.metadata.create_all(bind=engine)


# Dependency to get database session
def get_api_db():
    """
    Dependency that provides a database session.

    Yields:
        SessionLocal: A database session object.

    Ensures that the database session is closed after the request is processed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("sql_agent.sql_api:app", host="0.0.0.0", port=8000, reload=True)
