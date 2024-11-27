"""Unit tests for the gen_docstrings module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_app.doctool import (
    agent_executor,
    agent_with_chat_history,
    memory,
    model,
)


class TestGenDocstrings(unittest.TestCase):
    """Test cases for the gen_docstrings module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {"configurable": {"session_id": "test-session"}}

    def tearDown(self):
        """Clean up after each test method."""
        memory.clear()

    @patch("langchain_openai.ChatOpenAI")
    def test_model_initialization(self, mock_chat):
        """Test if the ChatOpenAI model is initialized correctly."""
        self.assertIsNotNone(model)
        mock_chat.assert_called_once_with(model="gpt-4o")

    def test_memory_initialization(self):
        """Test if the chat message history is initialized correctly."""
        self.assertEqual(memory.session_id, "test-session")
        self.assertEqual(len(memory.messages), 0)

    @patch("langchain.agents.AgentExecutor.invoke")
    def test_agent_execution(self, mock_invoke):
        """Test if the agent executes correctly with a sample input."""
        mock_invoke.return_value = {"output": "Sample docstring generated"}

        result = agent_executor.invoke({"input": "test_file.py"}, config=self.config)

        self.assertIn("output", result)
        self.assertEqual(result["output"], "Sample docstring generated")
        mock_invoke.assert_called_once()

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_agent_with_chat_history(self, mock_history_invoke):
        """Test if the agent with chat history handles input correctly."""
        mock_history_invoke.return_value = {"output": "Generated docstring"}

        result = agent_with_chat_history.invoke({"input": "sample.py"}, self.config)

        self.assertIn("output", result)
        self.assertEqual(result["output"], "Generated docstring")
        mock_history_invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
