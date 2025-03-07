"""
This is the main chatbot program/file for conversational capabilities and info distribution.
"""

import chainlit as cl

from indigobot.context import invoke_indybot

# from indigobot.quick_api import start_api
from indigobot.utils.etl.custom_loader import start_loader


def load():
    """
    Prompt user to execute the document loader functionality.

    Asks the user if they want to run the document loader and executes it if confirmed.
    Uses the start_loader() function from custom_loader module.

    :raises: Exception if the loader encounters an error
    """
    load_res = input("Would you like to execute the loader? (y/n) ")
    if load_res == "y":
        try:
            start_loader()
        except Exception as e:
            print(f"Error booting loader: {e}")


def api():
    """
    Prompt user to start the API server.

    Asks the user if they want to enable the API server and starts it if confirmed.
    Launches quick_api.py as a subprocess and waits 10 seconds for initialization.

    :raises: Exception if the API server fails to start
    """
    # load_res = input("Would you like to enable the API? (y/n) ")
    # if load_res == "y":
    #     try:
    #         api_thread = threading.Thread(target=start_api, daemon=True)
    #         api_thread.start()
    #     except Exception as e:
    #         print(f"Error booting API: {e}")


def main(cl_message: cl.Message) -> None:
    """
    Main function that runs the interactive chat loop.
    Initializes the chatbot environment and starts an interactive session.
    Handles user input and displays model responses in a loop until the user exits
    by entering an empty line.

    :param skip_loader: If True, skips the document loader prompt. Useful for testing.
    :type skip_loader: bool
    :param skip_api: If True, skips the API server prompt. Useful for testing.
    :type skip_api: bool
    :return: None
    :raises: KeyboardInterrupt if user interrupts with Ctrl+C
    :raises: Exception for any other runtime errors
    """

    if cl_message:
        # Configuration constants
        thread_config = {"configurable": {"thread_id": cl.context.session.id}}
        response = invoke_indybot(cl_message, thread_config=thread_config)
        if response:
            return response
        else:
            return "No response from chatbot!"

    # Prevents infinite loop when run directly
    return "No input received."


if __name__ == "__main__":
    main()
