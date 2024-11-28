"""
Configuration settings for the CfSS application.

This module contains path configurations and URL lists for data scraping operations.
All URL endpoints and file paths used across the application should be defined here.
"""

import os
from typing import Final, List

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from enum import Enum


class Models(Enum):
    GPT = ChatOpenAI(model="gpt-4o")
    GEMINI = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    CLAUDE = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Directory paths
CURRENT_DIR: Final[str] = os.path.dirname(__file__)
RAG_DIR: Final[str] = os.path.join(CURRENT_DIR, "src/indigobot", "rag_data")
CHROMA_DIR: Final[str] = os.path.join(RAG_DIR, ".chromadb")
GEM_DB: Final[str] = os.path.join(CHROMA_DIR, "gemini/chroma.sqlite3")
GPT_DB: Final[str] = os.path.join(CHROMA_DIR, "openai/chroma.sqlite3")

# OpenAI embeddings
try:
    gpt_vstore = Chroma(
        persist_directory=os.path.join(CHROMA_DIR, "openai"),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
except Exception as e:
    print(f"Error initializing OpenAI vectorstore: {e}")
    raise

# Google embeddings
try:
    gem_vstore = Chroma(
        persist_directory=os.path.join(CHROMA_DIR, "gemini"),
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_query"
        ),
    )
except Exception as e:
    print(f"Error initializing Google vectorstore: {e}")
    raise


class VectorStore(Enum):
    GPT = gpt_vstore
    GEM = gem_vstore


# URLs for API endpoints that return JSON data
urls: List[str] = [
    "https://rosecityresource.streetroots.org/api/query",  # Street Roots Resource API
]

# URLs for web pages that need recursive scraping
r_urls: List[str] = [
    "https://www.multco.us/food-assistance/get-food-guide",  # Food assistance guide
    "https://www.multco.us/dchs/rent-housing-shelter",  # Housing and shelter
    "https://www.multco.us/veterans",  # Veterans services
    "https://www.multco.us/dd",  # Developmental disabilities
]

# URL of the sitemap
sitemaps: List[str] = [
    "https://centralcityconcern.org/housing-sitemap.xml",
    "https://centralcityconcern.org/healthcare-sitemap.xml",
    "https://centralcityconcern.org/recovery-sitemap.xml",
    "https://centralcityconcern.org/jobs-sitemap.xml",
]
