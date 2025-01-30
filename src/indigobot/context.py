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

import sqlite3
import json
import hashlib
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import BaseMessage, HumanMessage, AIMessage

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
    def __init__(self, db_path="chat_cache.db"):
        """Initialize the SQLite cache database."""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create cache table if it doesn't exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            input_text TEXT,
            context TEXT,
            response TEXT
        )
        """)
        self.conn.commit()

    def _serialize_messages(self, messages):
        """Convert LangChain messages into JSON serializable format."""
        return json.dumps([{"type": type(msg).__name__, "content": msg.content} for msg in messages])

    def _deserialize_messages(self, messages_json):
        """Convert stored JSON messages back into LangChain message objects."""
        messages = json.loads(messages_json)
        deserialized_messages = []
        for msg in messages:
            if msg["type"] == "HumanMessage":
                deserialized_messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "AIMessage":
                deserialized_messages.append(AIMessage(content=msg["content"]))
            else:
                deserialized_messages.append(BaseMessage(content=msg["content"]))  # Fallback
        return deserialized_messages

    def get(self, input_text, cache_key, context):
    #Retrieve a cached response and deserialize it correctly.
        self.cursor.execute("SELECT response FROM cache WHERE cache_key = ?", (cache_key,))
        result = self.cursor.fetchone()
    
        if result:
            response_json = result[0]  # Ensure this is a JSON string
            try:
                return self._deserialize_messages(response_json)  # Convert back to message objects
            except json.JSONDecodeError:
                return [AIMessage(content=response_json)]  # Fallback to treating as raw text

        return None  # No cached result

    def set(self, input_text, cache_key, context, response):
        """Store a response in the cache, ensuring it's an AIMessage."""
        
        # Ensure response is wrapped in a message object
        if isinstance(response, dict) and "answer" in response:
            response = [AIMessage(content=response["answer"])]  # Convert to list of messages

        response_json = self._serialize_messages(response)  # Serialize response
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (cache_key, input_text, context, response) VALUES (?, ?, ?, ?)",
            (cache_key, input_text, context, response_json),
        )
        self.conn.commit()

    def clear_cache(self):
        """Clear all cached responses."""
        self.cursor.execute("DELETE FROM cache")
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

# Instantiate the cache
# create a sql database version
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
    Decorator to handle caching of model responses using SQLite.
    """
    def wrapper(state: ChatState):
        input_text = state["input"]

        # Generate a hashed cache key based on input_text and limited chat history
        chat_history = [
            message.content if isinstance(message, BaseMessage) else str(message)
            for message in state["chat_history"]
        ]
        context = str(state["context"])

        # Generate the hash of the cache key
        cache_key = hash_cache_key(input_text, chat_history, context)

        # Check cache first
        cached_response = chatbot_cache.get(input_text, cache_key, context)
        if cached_response:
            print(f"Cache hit for input: {input_text}")

            # Ensure cached response is formatted as a dictionary that matches ChatState
            return {
                "chat_history": list(state["chat_history"]) + cached_response,
                "context": context,
                "answer": cached_response[-1].content if cached_response else "",
            }
        else:
            print(f"Cache miss for input: {input_text}")
            response = func(state)  # Call the original function
            chatbot_cache.set(input_text, cache_key, context, response["chat_history"][-1:])
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
