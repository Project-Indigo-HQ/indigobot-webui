# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

    def __call__(self, *args, **kwargs):
        return Mock()


MOCK_MODULES = [
    "anthropic",
    "bs4",
    "chromadb",
    "chromadb.config",
    "google.generativeai",
    "langchain",
    "langchain.agents",
    "langchain.chains",
    "langchain.chains.combine_documents",
    "langchain.schema",
    "langchain.schema.language_model",
    "langchain.text_splitter",
    "langchain.tools",
    "langchain.tools.retriever",
    "langchain_anthropic",
    "langchain_chroma",
    "langchain_community",
    "langchain_community.agent_toolkits",
    "langchain_community.agent_toolkits.sql.base",
    "langchain_community.agent_toolkits.load_tools",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.recursive_url_loader",
    "langchain_community.document_transformers",
    "langchain_community.utilities",
    "langchain_core",
    "langchain_core.callbacks",
    "langchain_core.callbacks.manager",
    "langchain_core.chat_history",
    "langchain_core.documents",
    "langchain_core.documents.base",
    "langchain_core.language_models",
    "langchain_core.language_models.base",
    "langchain_core.load",
    "langchain_core.messages",
    "langchain_core.output_parsers",
    "langchain_core.output_parsers.base",
    "langchain_core.prompts",
    "langchain_core.pydantic_v1",
    "langchain_core.retrievers",
    "langchain_core.runnables",
    "langchain_core.runnables.base",
    "langchain_core.runnables.config",
    "langchain_core.runnables.history",
    "langchain_core.runnables.utils",
    "langchain_core.vectorstores",
    "langchain_core.tools",
    "langchain_core.tools.base",
    "langchain_core.utils",
    "langchain_core.utils.input",
    "langchain_core.utils.math",
    "langchain_core.utils.parallel",
    "langchain_experimental",
    "langchain_experimental.tools",
    "langchain_google_genai",
    "langchain_openai",
    "langgraph",
    "langgraph.checkpoint",
    "openai",
    "numpy",
    "pandas",
    "sqlalchemy",
    "unidecode",
]


# Create base class for language models
class BaseLLM:
    pass


class BaseLanguageModel:
    pass


# Add the base classes to the mock system
sys.modules["langchain_core.language_models.base"] = type(
    "langchain_core.language_models.base",
    (),
    {"BaseLanguageModel": BaseLanguageModel, "BaseLLM": BaseLLM},
)

# Update all mock modules
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Project Indigo"
copyright = "2024 - Kyle Klein, Avesta Mirashrafi, Melissa Shanks, Grace Trieu, Karl Rosenberg, JunFan Lin, Sam Nelson"
author = "Kyle Klein, Avesta Mirashrafi, Melissa Stokes, Grace Trieu, Karl Rosenberg, JunFan Lin, Sam Nelson"
version = "1.0.0"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_copybutton",
    "sphinx.ext.autodoc.typehints",
    "myst_parser",
]

# Add source file mappings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "langchain": ("https://api.python.langchain.com/en/latest/", None),
}
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# suppress_warnings = ["epub.unknown_project_files"]
