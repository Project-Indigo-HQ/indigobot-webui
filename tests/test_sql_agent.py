import json
import os
import sqlite3
import unittest
from unittest.mock import Mock, patch

from langchain_community.document_loaders import Document

from langchain_app.sql_agent import (
    load_docs,
    load_urls,
    format_docs,
    query_database,
    db_path,
)


class TestSQLAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.test_db_path = "test_indigo_bot_db.sqlite"
        # Temporarily override db_path
        global db_path
        cls.original_db_path = db_path
        db_path = cls.test_db_path

    def setUp(self):
        """Create fresh test database before each test"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                text TEXT,
                metadata TEXT
            );
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up test database after each test"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    @classmethod
    def tearDownClass(cls):
        """Restore original db_path"""
        global db_path
        db_path = cls.original_db_path

    def test_load_docs(self):
        """Test loading documents into database"""
        test_docs = [
            Document(
                page_content="Test content 1",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="Test content 2",
                metadata={"source": "test2.txt"}
            )
        ]
        
        load_docs(test_docs)
        
        results = query_database("SELECT text, metadata FROM documents")
        self.assertEqual(len(results), 2)
        self.assertIn("Test content 1", results[0][0])
        metadata1 = json.loads(results[0][1])
        self.assertEqual(metadata1["source"], "test1.txt")

    @patch('langchain_app.sql_agent.AsyncHtmlLoader')
    def test_load_urls(self, mock_loader):
        """Test loading URLs"""
        mock_docs = [
            Document(
                page_content="Web content 1",
                metadata={"source": "http://test1.com"}
            )
        ]
        mock_loader.return_value.load.return_value = mock_docs
        
        test_urls = ["http://test1.com"]
        load_urls(test_urls)
        
        results = query_database("SELECT text, metadata FROM documents")
        self.assertEqual(len(results), 1)
        self.assertIn("Web content 1", results[0][0])

    def test_format_docs(self):
        """Test document formatting"""
        test_docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2")
        ]
        
        result = format_docs(test_docs)
        self.assertEqual(result, "Content 1\n\nContent 2")

    def test_query_database(self):
        """Test database querying"""
        # Insert test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (text, metadata) VALUES (?, ?)",
            ("Test text", '{"source": "test.txt"}')
        )
        conn.commit()
        conn.close()
        
        results = query_database("SELECT text FROM documents")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "Test text")


if __name__ == '__main__':
    unittest.main()
