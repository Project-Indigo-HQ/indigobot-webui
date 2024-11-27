"""Unit tests for base_model.py"""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START

from langchain_app.base_model import State, call_model, workflow


class TestBaseModel(unittest.TestCase):
    """Test cases for base_model.py functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_state = {
            "input": "test question",
            "chat_history": [],
            "context": "test context",
            "answer": "",
        }

    def test_state_class(self):
        """Test State class initialization and typing"""
        self.assertEqual(self.test_state["input"], "test question")
        self.assertEqual(self.test_state["chat_history"], [])
        self.assertEqual(self.test_state["context"], "test context")
        self.assertEqual(self.test_state["answer"], "")

    @patch("langchain_app.base_model.rag_chain")
    def test_call_model(self, mock_rag_chain):
        """Test call_model function"""
        # Mock the rag_chain response
        mock_rag_chain.invoke.return_value = {
            "answer": "test answer",
            "context": "test context",
        }

        result = call_model(self.test_state)

        # Verify rag_chain was called with correct state
        mock_rag_chain.invoke.assert_called_once_with(self.test_state)

        # Check response structure
        self.assertIn("chat_history", result)
        self.assertIn("context", result)
        self.assertIn("answer", result)

        # Verify chat history format and content
        self.assertEqual(len(result["chat_history"]), 2)

        human_msg = result["chat_history"][0]
        ai_msg = result["chat_history"][1]

        self.assertIsInstance(human_msg, HumanMessage)
        self.assertIsInstance(ai_msg, AIMessage)
        self.assertEqual(str(human_msg.content), "test question")
        self.assertEqual(str(ai_msg.content), "test answer")
        self.assertEqual(result["context"], "test context")
        self.assertEqual(result["answer"], "test answer")

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
        from langchain_app.base_model import main

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
    unittest.main()
