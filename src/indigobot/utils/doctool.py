"""
A program that provides an AI agent for generating Sphinx-style docstrings, unit tests, and code summaries.

This module uses LangChain to create an AI agent that can analyze Python source files and generate:
- Sphinx-style docstrings for all functions and classes
- Unit tests compatible with the unittest framework  
- Detailed code summaries and suggestions

The agent assumes provided files are in the current working directory unless a full path is specified.

Example:
    To use the doctool from command line::

        $ python -m indigobot.utils.doctool
        Please enter a file path or name to generate docstrings (can use relative path/name):
        llm>> myfile.py

Note:
    The agent uses GPT models configured in the indigobot.config module.
"""

import readline  # Required for using arrow keys in CLI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.tools import PythonREPLTool

from indigobot.config import llm

memory = InMemoryChatMessageHistory(session_id="test-session")

instructions = """You are an expert at evaluating python programs and then writing 
comments, Sphinx-style docstrings, and unit tests to be used with the `unittest` suite. 
These tasks are your only job. Do not make an API call if asked to do anything else 
and instead ask the user to provide a file. You will write docstrings for all 
functions (including `main()`, if present) and classes defined in a file that is 
given to you. Also, provide a verbose general summary of the file and suggestions made.
The file given to you is in the current directory or uses a relative path from 
the current directory, unless explicitly specified.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = load_tools(["terminal"], allow_dangerous_tools=True)
tools.extend([PythonREPLTool()])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed;
    # It isn't really used here because we are using a simple in-memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}

print(
    "Please enter a file path or name to generate docstrings (can use relative path/name): "
)

while True:
    try:
        line = input("llm>> ")
        if line:
            result = agent_with_chat_history.invoke({"input": line}, config)["output"]
            print(result)
        else:
            break
    except Exception as e:
        print(e)
