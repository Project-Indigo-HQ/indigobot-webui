"""Unit tests for base_model.py"""

import pytest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START

from indigobot import State, call_model, workflow


@pytest.fixture
def test_state():
    """Test state fixture"""
    return {
        "input": "test question",
        "chat_history": [],
        "context": "test context", 
        "answer": "",
    }

@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain fixture"""
    with patch("langchain_app.base_model.rag_chain") as mock:
        mock.invoke.return_value = {
            "answer": "test answer",
            "context": "test context",
        }
        yield mock

def test_state_class(test_state):
    """Test State class initialization and typing"""
    assert test_state["input"] == "test question"
    assert test_state["chat_history"] == []
    assert test_state["context"] == "test context"
    assert test_state["answer"] == ""

def test_call_model(test_state, mock_rag_chain):
    """Test call_model function"""
    result = call_model(test_state)

    # Verify rag_chain was called with correct state
    mock_rag_chain.invoke.assert_called_once_with(test_state)

    # Check response structure
    assert "chat_history" in result
    assert "context" in result 
    assert "answer" in result

    # Verify chat history format and content
    assert len(result["chat_history"]) == 2

    human_msg = result["chat_history"][0]
    ai_msg = result["chat_history"][1]

    assert isinstance(human_msg, HumanMessage)
    assert isinstance(ai_msg, AIMessage)
    assert str(human_msg.content) == "test question"
    assert str(ai_msg.content) == "test answer"
    assert result["context"] == "test context"
    assert result["answer"] == "test answer"

    def test_workflow_structure(self):
        """Test workflow graph structure"""
        # Verify workflow has expected nodes
        self.assertIn("model", workflow.nodes)

        # Verify START edge exists
        edges = workflow.edges
        start_edges = [edge for edge in edges if edge[0] == START]
        self.assertTrue(any(edge[1] == "model" for edge in start_edges))

    @patch("builtins.input")
    @patch("langchain_app.base_model.app")
    @patch("langchain_app.base_model.retriever")
    def test_main_function(self, mock_retriever, mock_app, mock_input):
        """Test main function with skip_loader"""
        from indigobot.__main__ import main

        # Mock the retriever's vectorstore response
        mock_retriever.vectorstore.get.return_value = {
            "metadatas": [{"source": "test_source.pdf"}]
        }

        # Mock the app's invoke response
        mock_app.invoke.return_value = {"answer": "test response"}

        # Mock user input to exit after one iteration
        mock_input.side_effect = ["test input", ""]

        # Test that main runs without error when skip_loader is True
        try:
            main(skip_loader=True)
        except Exception as e:
            self.fail(f"main() raised {type(e).__name__} unexpectedly!")

        # Verify mocks were called correctly
        mock_app.invoke.assert_called_once()
        mock_retriever.vectorstore.get.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
