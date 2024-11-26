"""
Configuration settings for the Indigo-CfSS application.

This module contains path configurations and URL lists for data scraping operations.
All URL endpoints and file paths used across the application should be defined here.
"""

import os
from typing import Final, List

# Directory paths
CURRENT_DIR: Final[str] = os.path.dirname(__file__)
RAG_DIR: Final[str] = os.path.join(CURRENT_DIR, "..", "rag_data")
CHROMADB_DIR: Final[str] = os.path.join(RAG_DIR, ".chromadb")
PDF_DIR: Final[str] = os.path.join(RAG_DIR, "pdfs")

# URLs for API endpoints that return JSON data
URLS: Final[List[str]] = [
    "https://rosecityresource.streetroots.org/api/query",  # Street Roots Resource API
]

# URLs for web pages that need recursive scraping
R_URLS: Final[List[str]] = [
    "https://www.multco.us/food-assistance/get-food-guide",  # Food assistance guide
    "https://www.multco.us/dchs/rent-housing-shelter",      # Housing and shelter
    "https://www.multco.us/veterans",                       # Veterans services
    "https://www.multco.us/dd",                            # Developmental disabilities
]
