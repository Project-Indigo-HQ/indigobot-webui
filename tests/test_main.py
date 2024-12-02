"""Unit tests for __main__.py"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START

from indigobot.__main__ import call_model, workflow


@pytest.fixture
def test_state():
    """Test state fixture"""
    return {
        "messages": [HumanMessage(content="test question")],
        "chat_history": [],
        "context": "test context", 
        "answer": None
    }


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain fixture"""
    with patch("indigobot.__main__.rag_chain") as mock:
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


def test_workflow_structure():
    """Test workflow graph structure"""
    # Verify workflow has expected nodes
    assert "model" in workflow.nodes
    assert "should_continue" in workflow.nodes
    assert "human_input" in workflow.nodes

    # Verify START edge exists and connects to model
    edges = workflow.edges
    start_edges = [edge for edge in edges if edge[0] == START]
    assert any(edge[1] == "model" for edge in start_edges)

    # Verify conditional edges exist
    model_edges = [edge for edge in edges if edge[0] == "model"]
    assert len(model_edges) > 0


@patch("builtins.input")
@patch("indigobot.__main__.app")
@patch("indigobot.__main__.retriever") 
@patch("indigobot.__main__.custom_loader")
def test_main_function(mock_custom_loader, mock_retriever, mock_app, mock_input):
    """Test main function with skip_loader"""
    from indigobot.__main__ import main

    # Mock the retriever's vectorstore response
    mock_retriever.vectorstore.get.return_value = {
        "metadatas": [{"source": "test_source.pdf"}],
        "documents": ["test document"]
    }

    # Mock the app's invoke response
    mock_app.invoke.return_value = {
        "answer": "test response",
        "messages": [
            HumanMessage(content="test input"),
            AIMessage(content="test response")
        ],
        "chat_history": [],
        "context": "test context"
    }

    # Mock user input sequence
    mock_input.side_effect = ["test input", "quit"]

    # Test that main runs without error when skip_loader is True
    main(skip_loader=True)

    # Verify mocks were called correctly
    mock_app.invoke.assert_called()
    mock_retriever.vectorstore.get.assert_called()
    mock_custom_loader.main.assert_not_called()

    # Verify input was handled
    assert mock_input.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])
