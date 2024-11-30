from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from base_model import rag_chain, State

# Initialize Slack app with bot token and signing secret
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

@app.command("/ask")
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
        ).dict()
        
        # Get response from RAG chain
        response = rag_chain.invoke(state)
        
        # Send response back to Slack
        say(f"Q: {question}\nA: {response['answer']}")
        
    except Exception as e:
        say(f"Sorry, I encountered an error: {str(e)}")

@app.event("app_mention")
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
        ).dict()
        
        # Get response from RAG chain
        response = rag_chain.invoke(state)
        
        # Send response back to Slack
        say(f"Q: {question}\nA: {response['answer']}")
        
    except Exception as e:
        say(f"Sorry, I encountered an error: {str(e)}")

def start_slack_bot():
    """Start the Slack bot in Socket Mode"""
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    start_slack_bot()
