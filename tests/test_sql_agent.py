import json
import os
import sqlite3
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from langchain_app.sql_agent import (
    load_docs,
    load_urls,
    format_docs,
    query_database,
    DB_PATH,
)


class TestSQLAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.test_db_path = "test_indigo_bot_db.sqlite"
        # Temporarily override DB_PATH
        global DB_PATH
        cls.original_db_path = DB_PATH
        DB_PATH = cls.test_db_path

    def setUp(self):
        """Create fresh test database before each test"""
        # Always start with a fresh database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        # Create new database and table
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                text TEXT,
                metadata TEXT
            );
        """
        )
        conn.commit()
        conn.close()

        # Verify the table is empty
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        assert count == 0, f"Database should be empty but contains {count} records"
        conn.close()

    def tearDown(self):
        """Clean up test database after each test"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    @classmethod
    def tearDownClass(cls):
        """Restore original DB_PATH"""
        global DB_PATH
        DB_PATH = cls.original_db_path

    def test_load_docs(self):
        """Test loading documents into database"""
        test_docs = [
            Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test content 2", metadata={"source": "test2.txt"}),
        ]

        load_docs(test_docs)

        results = query_database("SELECT text, metadata FROM documents")
        self.assertEqual(len(results), 2)
        self.assertIn("Test content 1", results[0][0])
        metadata1 = json.loads(results[0][1])
        self.assertEqual(metadata1["source"], "test1.txt")

    @patch("langchain_app.sql_agent.AsyncHtmlLoader")
    def test_load_urls(self, mock_loader):
        """Test loading URLs"""
        mock_docs = [
            Document(
                page_content="Web content 1", metadata={"source": "http://test1.com"}
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
            Document(page_content="Content 2"),
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
            ("Test text", '{"source": "test.txt"}'),
        )
        conn.commit()
        conn.close()

        # Test normal query
        results = query_database("SELECT text FROM documents", params=())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "Test text")

    def test_query_database_with_params(self):
        """Test parameterized database querying"""
        # Insert test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (text, metadata) VALUES (?, ?)",
            ("Test text", '{"source": "test.txt"}'),
        )
        conn.commit()
        conn.close()

        # Test parameterized query
        results = query_database(
            "SELECT text FROM documents WHERE text = ?", params=("Test text",)
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "Test text")

    def test_query_database_injection_prevention(self):
        """Test SQL injection prevention"""
        # Insert test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (text, metadata) VALUES (?, ?)",
            ("Safe text", '{"source": "test.txt"}'),
        )
        conn.commit()
        conn.close()

        # Attempt SQL injection
        malicious_input = "' OR '1'='1"
        results = query_database(
            "SELECT text FROM documents WHERE text = ?", params=(malicious_input,)
        )
        self.assertEqual(len(results), 0)  # Should not match anything

    def test_load_docs_with_malicious_content(self):
        """Test handling of potentially malicious document content"""
        malicious_docs = [
            Document(
                page_content="'; DROP TABLE documents; --",
                metadata={"source": "malicious.txt"},
            )
        ]

        load_docs(malicious_docs)

        # Verify table still exists and data was properly escaped
        results = query_database(
            "SELECT text FROM documents WHERE text = ?",
            params=("'; DROP TABLE documents; --",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "'; DROP TABLE documents; --")


if __name__ == "__main__":
    unittest.main()
