import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document
from langchain_community.utilities import SQLDatabase

from indigobot.sql_agent.sql_agent import init_db, main


@pytest.fixture
def temp_db_path():
    """Fixture to create a temporary database file"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)  # Clean up after test


@pytest.fixture
def sample_docs():
    """Fixture to create sample documents"""
    return [
        Document(
            page_content="Test document 1",
            metadata={"source": "test1.txt", "page": 1},
        ),
        Document(
            page_content="Test document 2",
            metadata={"source": "test2.txt", "page": 2, "score": 0.95},
        ),
    ]


@pytest.fixture
def db_connection(temp_db_path):
    """Fixture to provide a database connection"""
    conn = sqlite3.connect(temp_db_path)
    yield conn
    conn.close()


class TestDatabase:
    def test_init_db(self, temp_db_path):
        """Test database initialization"""
        db = init_db(temp_db_path)
        assert isinstance(db, SQLDatabase)

        # Verify tables were created
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check if required tables exist
        for table in ["documents", "embedding_metadata"]:
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table,),
            )
            assert cursor.fetchone() is not None, f"Table {table} was not created"

        conn.close()

    def test_database_query(self, temp_db_path):
        """Test basic database querying"""
        db = init_db(temp_db_path)
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        conn.close()

    def test_init_db_with_invalid_path(self):
        """Test database initialization with invalid path"""
        with pytest.raises(Exception):
            init_db("/invalid/path/to/db.sqlite")

    def test_table_schema(self, temp_db_path):
        """Test that tables have correct schema"""
        db = init_db(temp_db_path)
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check documents table schema
        cursor.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        expected_columns = {
            "id": "INTEGER",
            "content": "TEXT",
            "key": "TEXT",
            "string_value": "TEXT",
            "int_value": "INTEGER",
            "float_value": "REAL",
            "bool_value": "BOOLEAN",
        }
        for col in columns:
            name, type_ = col[1], col[2]
            assert name in expected_columns
            assert expected_columns[name] in type_

        # Check embedding_metadata table schema
        cursor.execute("PRAGMA table_info(embedding_metadata)")
        columns = cursor.fetchall()
        expected_columns = {
            "id": "INTEGER",
            "document_id": "INTEGER",
            "embedding": "BLOB",
            "metadata": "TEXT",
        }
        for col in columns:
            name, type_ = col[1], col[2]
            assert name in expected_columns
            assert expected_columns[name] in type_

        conn.close()


class TestDocumentHandling:
    """Test class for document handling functionality"""

    def test_insert_document(self, temp_db_path):
        """Test inserting a document into the database"""
        db = init_db(temp_db_path)
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Insert test document
        cursor.execute(
            """
            INSERT INTO documents (content, key, string_value)
            VALUES (?, ?, ?)
            """,
            ("Test content", "test_key", "test_value"),
        )
        conn.commit()

        # Verify document was inserted
        cursor.execute("SELECT content FROM documents WHERE key = ?", ("test_key",))
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "Test content"

        conn.close()

    def test_document_metadata_relationship(self, temp_db_path):
        """Test relationship between documents and embedding_metadata"""
        db = init_db(temp_db_path)
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Insert test document
        cursor.execute(
            """
            INSERT INTO documents (content, key)
            VALUES (?, ?)
            """,
            ("Test content", "test_key"),
        )
        doc_id = cursor.lastrowid

        # Insert test metadata
        test_embedding = b"test_embedding_data"
        test_metadata = '{"source": "test"}'
        cursor.execute(
            """
            INSERT INTO embedding_metadata (document_id, embedding, metadata)
            VALUES (?, ?, ?)
            """,
            (doc_id, test_embedding, test_metadata),
        )
        conn.commit()

        # Verify relationship
        cursor.execute(
            """
            SELECT d.content, em.metadata 
            FROM documents d
            JOIN embedding_metadata em ON d.id = em.document_id
            WHERE d.key = ?
            """,
            ("test_key",),
        )
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "Test content"
        assert result[1] == test_metadata

        conn.close()


class TestMainFunction:
    def test_main_function_initialization(self):
        """Test main function initialization"""
        with patch("indigobot.utils.sql_agent.init_db") as mock_init_db, patch(
            "indigobot.utils.sql_agent.hub.pull"
        ) as mock_hub_pull, patch(
            "indigobot.utils.sql_agent.create_react_agent"
        ) as mock_create_agent, patch(
            "builtins.input", side_effect=["quit"]
        ):
            # Create a mock SQLDatabase
            mock_db = Mock(spec=SQLDatabase)
            mock_init_db.return_value = mock_db

            mock_prompt = Mock()
            mock_prompt.messages = [Mock()]
            mock_hub_pull.return_value = mock_prompt

            main()

            mock_init_db.assert_called_once()
            mock_hub_pull.assert_called_once_with(
                "langchain-ai/sql-agent-system-prompt"
            )
            mock_create_agent.assert_called_once()

    def test_main_function_with_query(self):
        """Test main function with a sample query"""
        with patch("indigobot.utils.sql_agent.init_db") as mock_init_db, patch(
            "indigobot.utils.sql_agent.hub.pull"
        ) as mock_hub_pull, patch(
            "indigobot.utils.sql_agent.create_react_agent"
        ) as mock_create_agent, patch(
            "builtins.input", side_effect=["Show tables", "quit"]
        ), patch(
            "builtins.print"
        ) as mock_print:
            # Setup mocks
            mock_db = Mock(spec=SQLDatabase)
            mock_init_db.return_value = mock_db

            mock_prompt = Mock()
            mock_prompt.messages = [Mock()]
            mock_hub_pull.return_value = mock_prompt

            mock_agent = Mock()
            mock_agent.stream.return_value = [
                {"messages": [Mock(content="Tables: documents, embedding_metadata")]}
            ]
            mock_create_agent.return_value = mock_agent

            main()

            # Verify agent was called with query
            assert mock_agent.stream.called
            mock_print.assert_called()

    def test_main_function_error_handling(self):
        """Test main function error handling"""
        with patch("indigobot.utils.sql_agent.init_db") as mock_init_db, patch(
            "builtins.print"
        ) as mock_print:
            # Simulate database initialization error
            mock_init_db.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                try:
                    main()
                except Exception:
                    mock_print.assert_called_with("Error: Database error")
                    raise
