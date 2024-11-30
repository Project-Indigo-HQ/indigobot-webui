import os

CURRENT_DIR = os.path.dirname(__file__)
RAG_DIR = os.path.join(CURRENT_DIR, "..", "rag_data")

# URL list for scraping JSON blob
URLS = [
    "https://rosecityresource.streetroots.org/api/query",
]

# URL list for recursively scraping
R_URLS = [
    "https://www.multco.us/food-assistance/get-food-guide",
    "https://www.multco.us/dchs/rent-housing-shelter",
    "https://www.multco.us/veterans",
    "https://www.multco.us/dd",
]
