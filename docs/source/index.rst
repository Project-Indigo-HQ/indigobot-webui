Project Indigo Documentation
============================

Welcome to Project Indigo's documentation. This project provides a sophisticated RAG (Retrieval-Augmented Generation) system using LangChain for intelligent document processing and retrieval.

Features
--------

* **Advanced RAG Implementation**: Uses state-of-the-art LLM models including GPT-4, Claude, and Gemini
* **Multi-Source Document Processing**: Handles PDFs, web pages, and structured data
* **Intelligent Web Crawling**: Built-in crawler with rate limiting and retry mechanisms
* **SQL Database Integration**: Native support for SQL databases with AI-powered querying
* **Vector Store Management**: Uses Chroma for efficient document embedding storage
* **Modular Architecture**: Easily extensible for custom document processors

Quick Start
-----------

To get started with Project Indigo:

.. code-block:: bash

   git clone https://github.com/yourusername/project-indigo
   cd project-indigo
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt

Configuration
-------------

The project requires several environment variables to be set:

* ``OPENAI_API_KEY``: Your OpenAI API key
* ``ANTHROPIC_API_KEY``: Your Anthropic API key (for Claude)
* ``GOOGLE_API_KEY``: Your Google API key (for Gemini)

Installation Requirements
-------------------------

* Python 3.12 or higher
* pip package manager
* Virtual environment (recommended)
* 16GB RAM recommended
* SSD storage recommended for vector database

Core Components
---------------

* **Document Loader**: Processes various document formats
* **Web Crawler**: Intelligently crawls and extracts web content
* **Vector Store**: Manages document embeddings
* **SQL Agent**: Handles database operations
* **RAG Engine**: Coordinates retrieval and generation

Usage Examples
--------------

Basic RAG Query:

.. code-block:: python

   from indigobot import main
   
   # Initialize the system
   main(skip_loader=False, skip_api=True)
   
   # Make a query
   response = main.query("What services are available?")
   print(response)

SQL Integration:

.. code-block:: python

   from indigobot.sql_agent import sql_agent
   
   # Initialize SQL agent
   agent = sql_agent.init_db()
   
   # Query the database
   result = agent.query("Show all available services")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   indigobot
   indigobot.utils
   indigobot.sql_agent

API Reference
-------------

See the :ref:`modindex` for detailed API documentation.

Contributing
------------

We welcome contributions! Please see our `Contributing Guidelines <https://github.com/yourusername/project-indigo/blob/main/CONTRIBUTING.md>`_ for more information.

License
-------

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
