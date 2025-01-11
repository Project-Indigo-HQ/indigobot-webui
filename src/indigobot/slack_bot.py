from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
import os
from base_model import rag_chain, State

# Initialize Flask app
flask_app = Flask(__name__)

# Initialize Slack app with bot token and signing secret
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Initialize the SlackRequestHandler
handler = SlackRequestHandler(slack_app)

@slack_app.command("/ask")
def handle_ask_command(ack, command, say):
    """Handle /ask command in Slack"""
    ack()
    question = command["text"]
    
    if not question.strip():
        say("Please provide a question after the /ask command")
        return
        
    try:
        # Initialize state with empty chat history
        state = State(
            input=question,
            chat_history=[],
            context=""
        ).model_dump()
        
        # Get response from RAG chain
        response = rag_chain.invoke(state)
        
        # Send response back to Slack
        say(f"Q: {question}\nA: {response['answer']}")
        
    except Exception as e:
        error_message = str(e)
        if "not_in_channel" in error_message:
            say("I need to be invited to this channel first! Please add me using /invite @YourBotName")
        else:
            say(f"Sorry, I encountered an error: {error_message}")

@slack_app.event("app_mention")
def handle_mention(event, say):
    """Handle when the bot is mentioned in a channel"""
    try:
        # Extract the question (remove the bot mention)
        text = event["text"]
        question = text.split(">", 1)[1].strip()
        
        if not question:
            say("Please provide a question when mentioning me")
            return
            
        # Initialize state with empty chat history
        state = State(
            input=question,
            chat_history=[],
            context=""
        ).model_dump()
        
        # Get response from RAG chain
        response = rag_chain.invoke(state)
        
        # Send response back to Slack
        say(f"Q: {question}\nA: {response['answer']}")
        
    except Exception as e:
        say(f"Sorry, I encountered an error: {str(e)}")

# Flask routes for Slack events
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events"""
    return handler.handle(request)

@flask_app.route("/slack/commands", methods=["POST"])
def slack_commands():
    """Handle Slack commands"""
    return handler.handle(request)

if __name__ == "__main__":
    # Run the Flask app on all interfaces
    flask_app.run(
        host='0.0.0.0', 
        port=8000
    )
