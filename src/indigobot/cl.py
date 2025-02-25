""" This module handles Chainlit Web UI integration"""

import chainlit as cl
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
import tempfile
from langchain.schema import AIMessage, HumanMessage
from indigobot.config import llm, vectorstore
from indigobot.context import call_model
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize pygame for audio playback
pygame.mixer.init()

# Create FastAPI app
app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    logging.debug("Serving index.html")
    with open("src/static/index.html", "r") as f:
        return f.read()
    
@app.post("/send_voice")
async def handle_voice(request: Request):
    """Receives voice input and sends it to the chatbot"""
    data = await request.json()
    text = data.get("text", "")

    if text:
        response = call_model({"input": text, "chat_history": [], "context": "", "answer": ""})
        bot_reply = response["answer"]
        text_to_speech(bot_reply)
        return JSONResponse(content={"response": bot_reply})
    
    return JSONResponse(content={"error": "No text received"}, status_code=400)

def text_to_speech(text):
    """Converts text to speech and plays the audio."""
    
    # Generate speech audio
    tts = gTTS(text)
    
    # Create a temporary file for cross-platform compatibility
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio_path = temp_audio.name
        tts.save(temp_audio_path)

    # Play the audio using pygame
    pygame.mixer.music.load(temp_audio_path)
    pygame.mixer.music.play()

    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Cleanup: Remove the temporary file after playback
    os.remove(temp_audio_path)

    return temp_audio_path  # Returning path just in case it's needed


# Function to convert speech to text
def speech_to_text():
    """Capture voice input and return the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for input...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # 5s timeout
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError:
            print("Error with speech recognition service.")
    return ""


@cl.on_message
async def on_message(message: cl.Message):
    """Handles user messages in the Chainlit chat."""
    
    # Handle STT
    if message.content.strip().lower() == "voice":
        message.content = speech_to_text() or "Sorry, I didn't catch that."

    # Construct state for chatbot
    state = {
        "input": message.content,
        "chat_history": [],
        "context": "",
        "answer": "",
    }
    
    # Get chatbot response
    response = call_model(state)
    bot_reply = response["answer"]
    
    # Send response in chat
    await cl.Message(bot_reply).send()
    
    # Convert bot response to speech
    text_to_speech(bot_reply)


if __name__ == "__main__":
    cl.run()
