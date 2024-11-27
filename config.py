"""
Configuration settings for the CfSS application.

This module contains path configurations and URL lists for data scraping operations.
All URL endpoints and file paths used across the application should be defined here.
"""

import os
from typing import Final, List

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI

LLMS = {
    "gpt": ChatOpenAI(model="gpt-4"),
    "gemini": GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0),
    "claude": ChatAnthropic(model="claude-3-5-sonnet-latest"),
}

# Directory paths
CURRENT_DIR: Final[str] = os.path.dirname(__file__)
RAG_DIR: Final[str] = os.path.join(CURRENT_DIR, "..", "rag_data")
CHROMA_DIR: Final[str] = os.path.join(RAG_DIR, ".chromadb")
GEM_DB: Final[str] = os.path.join(CHROMA_DIR, "gemini/chroma.sqlite3")
OAI_DB: Final[str] = os.path.join(CHROMA_DIR, "openai/chroma.sqlite3")
PDF_DIR: Final[str] = os.path.join(RAG_DIR, "pdfs")

# URLs for API endpoints that return JSON data
URLS: Final[List[str]] = [
    "https://rosecityresource.streetroots.org/api/query",  # Street Roots Resource API
]

# URLs for web pages that need recursive scraping
R_URLS: Final[List[str]] = [
    "https://www.multco.us/food-assistance/get-food-guide",  # Food assistance guide
    "https://www.multco.us/dchs/rent-housing-shelter",  # Housing and shelter
    "https://www.multco.us/veterans",  # Veterans services
    "https://www.multco.us/dd",  # Developmental disabilities
]
