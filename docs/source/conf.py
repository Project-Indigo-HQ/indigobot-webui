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
    "openai",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "anthropic",
    "google.generativeai",
    "langchain.schema",
    "langchain.schema.language_model",
    "langchain_core.language_models.base",
    "langchain_core.language_models",
    "langchain_core.pydantic_v1",
    "langchain.agents",
    "langchain_community.agent_toolkits",
    "langchain_community.agent_toolkits.sql.base",
    "langchain_community.utilities",
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

suppress_warnings = ["epub.unknown_project_files"]
