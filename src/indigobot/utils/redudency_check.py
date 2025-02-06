import os
import json
import importlib

def check_duplicate(base_url,urls):
    """
    Check if the URL is already loded

    param base_url: List of base URL to check
    param url: URL to check

    return: Return urls that are not loaded
    """
    urls_to_load = []
    for url in urls:
        if url in base_url:
            continue
        else:
            urls_to_load.append(url)
    return urls_to_load            

def traking_urls_update(new_urls):
    """
    Update the tracking URL list with the new URLs
    """
    # Dynamically import the config module
    config = importlib.import_module('indigobot.config')
    
    for url in new_urls:
        if url not in config.tracked_urls:
            config.tracked_urls.append(url)
    
    # Construct the path to config.py using CURRENT_DIR from the imported config module
    config_path = os.path.join(config.CURRENT_DIR, "config.py")
    
    # Read the current content of config.py
    with open(config_path, "r") as file:
        config_content = file.readlines()
    
    # Find the line where tracked_urls is defined
    start_idx = None
    end_idx = None
    for i, line in enumerate(config_content):
        if line.startswith("tracked_urls ="):
            start_idx = i
        elif start_idx is not None and line.startswith("]"):
            end_idx = i
            break
    
    # Update the tracked_urls list in the config content
    if start_idx is not None and end_idx is not None:
        updated_tracked_urls = json.dumps(config.tracked_urls, indent=4)
        config_content[start_idx:end_idx + 1] = [f"tracked_urls = {updated_tracked_urls}\n"]
    
    # Write the updated content back to config.py
    with open(config_path, "w") as file:
        file.writelines(config_content)