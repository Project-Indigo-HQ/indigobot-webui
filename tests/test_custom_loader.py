import unittest
from unittest.mock import MagicMock, patch

from langchain.schema import Document

from indigobot.utils.custom_loader import (
    chunking,
    clean_documents,
    clean_text,
    extract_text,
    scrape_main,
)


class TestCustomLoader(unittest.TestCase):
    def test_clean_text(self):
        """Test clean_text function with various inputs"""
        test_cases = [
            ("Hello  World", "Hello World"),  # Extra spaces
            ("Café", "Cafe"),  # Unicode characters
            ("\n\tTest\n", "Test"),  # Whitespace characters
            ("Multiple     Spaces", "Multiple Spaces"),  # Multiple spaces
            ("", ""),  # Empty string
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

    def test_chunking_mock(self):
        """Test chunking function with mocks"""
        with patch(
            "indigobot.utils.custom_loader.RecursiveCharacterTextSplitter"
        ) as mock_splitter:
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

    def test_chunking_real(self):
        """Test chunking function with real documents"""
        # Create a test document with content longer than chunk size
        long_text = " ".join(["word"] * 20000)  # Create much longer text
        docs = [Document(page_content=long_text, metadata={})]

        chunks = chunking(docs)

        # Verify chunks were created properly
        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 10000)  # Check chunk sizes

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

    @patch("indigobot.utils.custom_loader.RecursiveUrlLoader")
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


if __name__ == "__main__":
    unittest.main()
