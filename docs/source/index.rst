Project Indigo Documentation
===========================

Welcome to Project Indigo's documentation. This project provides a sophisticated RAG (Retrieval-Augmented Generation) system using LangChain.

Quick Start
----------

To get started with Project Indigo:

.. code-block:: bash

   git clone https://github.com/yourusername/project-indigo
   cd project-indigo
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt

Installation
-----------

Requirements:

* Python 3.12 or higher
* pip package manager
* Virtual environment (recommended)

For development installation:

.. code-block:: bash

   pip install -r requirements-dev.txt

Features
--------

* Custom document loading and processing
* Advanced RAG implementation
* SQL database integration
* Web crawling capabilities

Components
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

API Reference
------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   langchain_app.base_model
   langchain_app.custom_loader
   langchain_app.sql_agent
   langchain_app.CCC_scraper.crawler
   langchain_app.CCC_scraper.refine_html

Contributing
-----------

We welcome contributions! Please see our contributing guidelines for more information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

