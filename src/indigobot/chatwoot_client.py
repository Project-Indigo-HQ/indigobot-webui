import requests
import os

CHATWOOT_API_URL = os.getenv("CHATWOOT_API_URL", "https://your-chatwoot-instance.com")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "your_api_token_here")

def send_message_to_chatwoot(conversation_id, message):
    url = f"{CHATWOOT_API_URL}/api/v1/conversations/{conversation_id}/messages"
    headers = {
        "Authorization": f"Bearer {CHATWOOT_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "content": message,
        "message_type": "outgoing"
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("Message sent back to Chatwoot successfully.")
    else:
        print(f"Failed to send message back to Chatwoot: {response.status_code} {response.text}")
