import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from langchain_app.custom_loader import (
    clean_text,
    clean_documents,
    chunking,
    extract_text,
    scrape_main,
    scrape_articles,
    PDF_PATH,
)

# PARENT_DIR = os.path.dirname("..")
# PDF_PATH = Path(
#     os.path.join(
#         PARENT_DIR, "langchain_app/rag_data/pdfs/NavigatingLLMsBegginersGuide.pdf"
#     )
# )


class TestCustomLoader(unittest.TestCase):
    def test_clean_text(self):
        """Test clean_text function with various inputs"""
        test_cases = [
            ("Hello  World", "Hello World"),  # Extra spaces
            ("Café", "Cafe"),  # Unicode characters
            ("\n\tTest\n", "Test"),  # Whitespace characters
            ("Multiple     Spaces", "Multiple Spaces"),  # Multiple spaces
        ]
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(clean_text(input_text), expected)

    def test_clean_documents(self):
        """Test clean_documents function"""
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Hello  World"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Café\n"

        docs = [mock_doc1, mock_doc2]
        cleaned_docs = clean_documents(docs)

        self.assertEqual(cleaned_docs[0].page_content, "Hello World")
        self.assertEqual(cleaned_docs[1].page_content, "Cafe")

    @patch("langchain_app.custom_loader.RecursiveCharacterTextSplitter")
    def test_chunking(self, mock_splitter):
        """Test chunking function"""
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_docs = [mock_doc]
        mock_chunks = [MagicMock()]
        mock_splitter_instance.split_documents.return_value = mock_chunks

        result = chunking(mock_docs)

        mock_splitter.assert_called_once_with(chunk_size=10000, chunk_overlap=1000)
        mock_splitter_instance.split_documents.assert_called_once_with(mock_docs)
        self.assertEqual(result, mock_chunks)

    def test_extract_text(self):
        """Test extract_text function"""
        html_content = """
        <html>
            <body>
                <div id="main">
                    <p>Test content</p>
                </div>
            </body>
        </html>
        """
        result = extract_text(html_content)
        self.assertEqual(result, "Test content")

        # Test with no main div
        html_content_no_main = """
        <html>
            <body>
                <p>Other content</p>
            </body>
        </html>
        """
        result = extract_text(html_content_no_main)
        self.assertEqual(result, "Other content")

    @patch("langchain_app.custom_loader.RecursiveUrlLoader")
    def test_scrape_main(self, mock_loader):
        """Test scrape_main function"""
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_docs = [mock_doc]
        mock_loader_instance.load.return_value = mock_docs

        result = scrape_main("http://example.com", 2)

        mock_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        self.assertEqual(result, mock_docs)

    @patch("langchain_app.custom_loader.AsyncHtmlLoader")
    @patch("langchain_app.custom_loader.BeautifulSoupTransformer")
    def test_scrape_articles(self, mock_transformer, mock_loader):
        """Test scrape_articles function"""
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance

        mock_docs = [MagicMock()]
        mock_loader_instance.load.return_value = mock_docs
        mock_transformed_doc = MagicMock()
        mock_transformed_doc.page_content = "Test content"
        mock_transformed_docs = [mock_transformed_doc]
        mock_transformer_instance.transform_documents.return_value = (
            mock_transformed_docs
        )

        result = scrape_articles(["http://example.com"])

        mock_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        mock_transformer_instance.transform_documents.assert_called_once_with(
            documents=mock_docs, tags_to_extract=[]
        )
        self.assertEqual(result, mock_transformed_docs)

    def test_pdf_path_exists(self):
        """Test that the PDF file path is valid"""
        self.assertTrue(os.path.exists(PDF_PATH), f"PDF file not found at {PDF_PATH}")
        self.assertTrue(
            str(PDF_PATH).find("/langchain_app/rag_data/"),
            "PDF path should be relative to langchain_app directory",
        )


if __name__ == "__main__":
    unittest.main()
