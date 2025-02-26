import chainlit as cl
import asyncio
from indigobot.__main__ import main as indybot

@cl.on_message
async def main(message: cl.Message):
    """Handle user input and send response from chatbot."""
    
    indybot_res = await asyncio.to_thread(indybot, message.content)

    msg = cl.Message(content=f"{indybot_res}")
    await msg.send()

