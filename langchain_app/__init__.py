"""
LangChain application for Social Services Chatbot (Indigo-CfSS).

This package provides a RAG-based chatbot that helps users find and access
social services information. It includes web scraping, document processing,
and a conversational interface powered by various LLM providers.

Author: CfSS Development Team
License: MIT
"""

from typing import List

__author__ = "CfSS Development Team"
__license__ = "MIT"

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0"  # package is not installed

# Explicitly declare public exports
__all__: List[str] = [
    "__version__",
    "__author__",
    "__license__",
]
