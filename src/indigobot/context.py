"""
Context management module for the chatbot application.

This module handles the chat context, state management, and conversation flow using LangChain
and LangGraph components. It maintains chat history, processes queries through a RAG
(Retrieval Augmented Generation) pipeline, and manages the conversational state.

The module integrates various components:
- LangChain for RAG operations and chat history management
- LangGraph for workflow management
- Custom state typing for type safety
"""

import readline  # Required for using arrow keys in CLI
from typing import Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from indigobot.config import llm, vectorstore

from hashlib import sha256
from functools import lru_cache
import hashlib
from langchain_core.messages import AIMessage, HumanMessage

chatbot_retriever = vectorstore.as_retriever()


class ChatState(TypedDict):
    """
    A dictionary that represents the state of a chat interaction/history.
    """

    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

class ChatbotCache:
    def __init__(self):
        self.cache = {}

    def get(self, input_text: str, chat_history: tuple, context: str):
        key = (input_text, tuple(chat_history), context)  # Convert chat_history to tuple for hashability
        return self.cache.get(key, None)

    def set(self, input_text: str, chat_history: tuple, context: str, response: dict):
        key = (input_text, tuple(chat_history), context)  # Convert chat_history to tuple for hashability
        self.cache[key] = response


# Instantiate the cache
chatbot_cache = ChatbotCache()

def normalize_text(text):
    """
    Normalize a text string by stripping whitespace and converting to lowercase.
    """
    if isinstance(text, str):
        return text.strip().lower()
    return str(text).strip().lower()

def hash_cache_key(input_text, chat_history, context):
    """
    Generate a hash of the cache key based on normalized input_text, chat_history, and context.
    """
    # Normalize input_text
    normalized_input = normalize_text(input_text)

    # Normalize chat history (last 2 messages)
    normalized_history = tuple(
        normalize_text(str(message)) for message in chat_history[-2:]
    )

    # Normalize context
    normalized_context = normalize_text(context)

    # Create a key string to hash
    key_string = f"{normalized_input}|{normalized_history}|{normalized_context}"

    # Return a hashed version of the key
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

# Cache Decorator with hashed keys
def cache_decorator(func):
    """
    Decorator to handle caching of model responses with normalized keys.
    """
    def wrapper(state: ChatState):
        input_text = normalize_text(state["input"])
        
        # Normalize chat history
        chat_history = [
            normalize_text(message.content if isinstance(message, BaseMessage) else str(message))
            for message in state["chat_history"]
        ]
        
        # Normalize context
        context = normalize_text(state["context"])
        
        # Generate the hash of the normalized cache key
        cache_key = hash_cache_key(input_text, chat_history, context)
        
        # Check cache first
        cached_response = chatbot_cache.get(input_text, cache_key, context)
        if cached_response:
            print(f"Cache hit for input: {input_text}")  # Logging cache hit
            return cached_response
        else:
            print(f"Cache miss for input: {input_text}")  # Logging cache miss
            response = func(state)  # Call the original function (i.e., call_model)
            chatbot_cache.set(input_text, cache_key, context, response)  # Cache the result
            return response

    return wrapper

# Updated cached_model_call to use the hashed key for caching
def cached_model_call(input_text: str, chat_history: list, context: str):
    """
    Cached version of the model call to reduce redundant API requests.

    :param input_text: The user's input.
    :type input_text: str
    :param chat_history: The chat history as a list of message objects (e.g., HumanMessage, AIMessage).
    :type chat_history: list
    :param context: The current context of the conversation.
    :type context: str
    :return: The model's response as a dictionary.
    :rtype: dict
    """
    # Prepare the state for the model call
    state = {
        "input": input_text,
        "chat_history": chat_history,
        "context": context,
        "answer": "",
    }

    # Call the model and return the response
    return chatbot_rag_chain.invoke(state)

@cache_decorator
def call_model(state: ChatState):
    """
    Call the language model with caching and return the updated state.
    """
    input_text = normalize_text(state["input"])

    # Normalize chat history
    chat_history = [
        HumanMessage(normalize_text(message.content)) if isinstance(message, BaseMessage) else HumanMessage(normalize_text(str(message)))
        for message in state["chat_history"]
    ]

    # Normalize context
    context = normalize_text(state["context"])

    # Call the cached model function
    try:
        response = cached_model_call(input_text, chat_history, context)
    except Exception as e:
        raise ValueError(f"Error in cached_model_call: {e}") from e

    # Update chat history and return the updated state
    updated_chat_history = list(state["chat_history"]) + [
        HumanMessage(input_text),
        AIMessage(response["answer"]),
    ]

    return {
        "chat_history": updated_chat_history,
        "context": response["context"],
        "answer": response["answer"],
    }

# Prompt configuration for question contextualization
contextualize_q_system_prompt = (
    "Reformulate the user's question into a standalone question, "
    "considering the chat history. Return the original question if no reformulation needed."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, chatbot_retriever, contextualize_q_prompt
)

# Prompt configuration for answer generation
system_prompt = (
    "You are an assistant that answers questions/provides information about "
    "social services in Portland, Oregon. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, "
    "say that you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
chatbot_rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

workflow = StateGraph(state_schema=ChatState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
chatbot_app = workflow.compile(checkpointer=memory)
