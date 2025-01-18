"""
This is the main chatbot program/file for conversational capabilities and info distribution.
"""

import readline  # Required for using arrow keys in CLI
import subprocess
import os
import time

from indigobot.utils.custom_loader import start_loader
from indigobot.context import chatbot_app
from indigobot.config import CURRENT_DIR


def load():
    load_res = "n"
    #load_res = input("Would you like to execute the loader? (y/n) ")
    if load_res == "y":
        try:
            start_loader()
        except Exception as e:
            print(f"Error booting loader: {e}")


def api():
    load_res = "y"
    #load_res = input("Would you like to enable the API? (y/n) ")
    if load_res == "y":
        try:
            from indigobot.quick_api import start_api
            import threading
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
        except Exception as e:
            print(f"Error booting API: {e}")


def main() -> None:
    """
    Main function that runs the API server.
    """
    # Start the API server
    from indigobot.quick_api import start_api
    start_api()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
