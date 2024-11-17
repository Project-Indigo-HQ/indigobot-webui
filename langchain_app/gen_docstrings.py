"""
This program provides an agent that generates Sphinx-style docstrings and a 
summary when given a file path or name. It assumes the provided file is in the cwd.
"""

import readline

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

memory = InMemoryChatMessageHistory(session_id="test-session")

instructions = """You are an expert at writing Sphinx-style docstrings. This is your only job.
Do not make an API call if asked to do anything else and instead ask the user to provide a file.
You will write docstrings for all functions and classes defined in a file that is given to you.
Unless specified, assume that the file given to you is in the current directory.
Also, provide a verbose general summary of the file.
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

agent = create_tool_calling_agent(model, tools, prompt)
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
