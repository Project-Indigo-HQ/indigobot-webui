import os
import pytest
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup
from langchain_app.custom_loader import (
    clean_text,
    clean_documents,
    chunking,
    extract_text,
)
from langchain.schema import Document

def test_clean_text():
    """Test the clean_text function with various inputs"""
    # Test basic whitespace cleaning
    assert clean_text("  hello   world  ") == "hello world"
    
    # Test unicode character replacement
    assert clean_text("héllo wörld") == "hello world"
    
    # Test multiple spaces
    assert clean_text("hello     world") == "hello world"
    
    # Test empty string
    assert clean_text("") == ""

def test_clean_documents():
    """Test the clean_documents function"""
    # Create test documents
    docs = [
        Document(page_content="  héllo   wörld  ", metadata={}),
        Document(page_content="test    document", metadata={})
    ]
    
    cleaned_docs = clean_documents(docs)
    
    assert cleaned_docs[0].page_content == "hello world"
    assert cleaned_docs[1].page_content == "test document"

def test_chunking():
    """Test the chunking function"""
    # Create a test document with content longer than chunk size
    long_text = " ".join(["word"] * 20000)  # Create much longer text
    docs = [Document(page_content=long_text, metadata={})]
        
    chunks = chunking(docs)
        
    assert len(chunks) > 1  # Should split into multiple chunks
    assert all(len(chunk.page_content) <= 10000 for chunk in chunks)  # Check chunk sizes

def test_extract_text():
    """Test the extract_text function"""
    # Test HTML with div#main
    html_with_main = """
    <html>
        <body>
            <div id="main">Main content here</div>
            <div>Other content</div>
        </body>
    </html>
    """
    assert extract_text(html_with_main).strip() == "Main content here"
    
    # Test HTML without div#main
    html_without_main = """
    <html>
        <body>
            <div>First content</div>
            <div>Second content</div>
        </body>
    </html>
    """
    assert extract_text(html_without_main).strip() == "First content Second content"

@pytest.fixture
def mock_vectorstore():
    """Fixture for mocking vectorstore"""
    return Mock()

@pytest.fixture
def sample_chunks():
    """Fixture for sample document chunks"""
    return [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]
