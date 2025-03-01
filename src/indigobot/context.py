"""
This module provides functionality for managing conversational state, caching responses,
and processing queries through a RAG (Retrieval Augmented Generation) pipeline.
"""

import readline  # Required for using arrow keys in CLI
from typing import Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated, TypedDict

from indigobot.config import llm, vectorstore

chatbot_retriever = vectorstore.as_retriever()


retriever_tool = create_retriever_tool(
    chatbot_retriever,
    "retrieve_documents",
    "Search and return information about documents as inquired by user.",
)

tools = [retriever_tool]

# Prompt configuration for answer generation
system_prompt = (
    "You are an assistant that answers questions/provides information about "
    "social services in Portland, Oregon. Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know. Or, if you don't have "
    "specific details about a place such as operating hours, use `lookup_place_tool` to "
    "inform your response. Do not mention to the user if you are missing info, "
    "just provide them with the info they asked for."
    "Use three sentences maximum and keep the answer concise."
)

memory = MemorySaver()
chatbot_app = create_react_agent(
    llm, tools=tools, prompt=system_prompt, checkpointer=memory
)
