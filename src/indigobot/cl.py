""" This module handles Chainlit Web UI integration"""

import chainlit as cl
from indigobot.__main__ import main as indybot


@cl.on_message
async def main(message: cl.Message):
    # Send user input from chainlit Web UI to propbot LLM program
    indybot_res = indybot(message.content)

    # Send propbot response back to the user Web UI
    await cl.Message(
        content=f"{indybot_res}",
    ).send()
