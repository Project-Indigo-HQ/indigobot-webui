import json
import os
import sqlite3
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from indigobot.config import GPT_DB
from indigobot.utils.sql_agent import format_docs, load_docs, load_urls, query_database


class TestSQLAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        # Store original DB path
        cls.original_db_path = GPT_DB
        cls.test_db_path = "test_indigo_bot_db.sqlite"
        # Remove existing test database if it exists
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

        # Initialize the test database
        conn = sqlite3.connect(cls.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                metadata TEXT
            );
        """
        )
        conn.commit()
        conn.close()

    def setUp(self):
        """Create fresh test database before each test"""
        # Clear all data before each test
        if os.path.exists(self.test_db_path):
            try:
                conn = sqlite3.connect(self.test_db_path, timeout=30)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
            finally:
                conn.close()

    def tearDown(self):
        """Clean up test database after each test"""
        if os.path.exists(self.test_db_path):
            try:
                conn = sqlite3.connect(self.test_db_path, timeout=30)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
            finally:
                conn.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up test database and restore original GPT_DB"""
        global GPT_DB
        GPT_DB = cls.original_db_path

        # Remove test database
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except OSError:
                pass

    def test_load_docs(self):
        """Test loading documents into database"""
        test_docs = [
            Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test content 2", metadata={"source": "test2.txt"}),
        ]

        load_docs(test_docs, db_path=self.test_db_path)

        results = query_database(
            "SELECT text, metadata FROM documents ORDER BY id",
            db_path=self.test_db_path,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "Test content 1")
        metadata1 = json.loads(results[0][1])
        self.assertEqual(metadata1["source"], "test1.txt")

    @patch("indigobot.utils.sql_agent.AsyncHtmlLoader")
    def test_load_urls(self, mock_loader):
        """Test loading URLs"""
        mock_docs = [
            Document(
                page_content="Web content 1", metadata={"source": "http://test1.com"}
            )
        ]
        mock_loader.return_value.load.return_value = mock_docs

        test_urls = ["http://test1.com"]
        load_urls(test_urls, db_path=self.test_db_path)

        results = query_database(
            "SELECT text, metadata FROM documents", db_path=self.test_db_path
        )
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
        results = query_database(
            "SELECT text FROM documents", params=(), db_path=self.test_db_path
        )
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
            "SELECT text FROM documents WHERE text = ?",
            params=("Test text",),
            db_path=self.test_db_path,
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
            "SELECT text FROM documents WHERE text = ?",
            params=(malicious_input,),
            db_path=self.test_db_path,
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

        load_docs(malicious_docs, db_path=self.test_db_path)

        # Verify table still exists and data was properly escaped
        results = query_database(
            "SELECT text FROM documents WHERE text = ?",
            params=("'; DROP TABLE documents; --",),
            db_path=self.test_db_path,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "'; DROP TABLE documents; --")


if __name__ == "__main__":
    unittest.main()
