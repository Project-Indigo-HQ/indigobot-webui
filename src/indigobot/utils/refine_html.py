import json
import os

from bs4 import BeautifulSoup
from langchain.schema import Document

from indigobot.config import RAG_DIR


def load_html_files(folder_path):
    """
    Load all HTML files from a specified directory.

    Scans the given directory and returns paths to all files with .html extension.
    Does not search subdirectories.

    :param folder_path: Path to the directory containing HTML files
    :type folder_path: str
    :return: List of absolute paths to HTML files
    :rtype: list[str]
    :raises OSError: If the folder_path doesn't exist or isn't accessible
    :raises TypeError: If folder_path is not a string

    Example:
        >>> html_files = load_html_files('/path/to/html/files')
        >>> print(html_files)
        ['/path/to/html/files/page1.html', '/path/to/html/files/page2.html']
    """
    html_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            html_files.append(file_path)
    return html_files


def parse_and_save(file_path):
    """
    Parse an HTML file to extract the title and headers, and save the result as a JSON file.

    Processes an HTML file by:
    1. Reading and parsing the HTML content
    2. Extracting the page title
    3. Finding all header tags (h1-h6) and paragraphs
    4. Saving the structured data as JSON in the processed_text directory

    :param file_path: Path to the HTML file to be parsed
    :type file_path: str
    :return: None
    :raises FileNotFoundError: If the input file doesn't exist
    :raises OSError: If there are issues reading the file or creating output directory
    :raises Exception: If HTML parsing fails or JSON serialization fails

    The output JSON structure follows this format:
    {
        "title": "Page Title",
        "headers": [
            {
                "tag": "h1",
                "text": "Header Text",
                "html": "<h1>Header Text</h1>"
            },
            ...
        ]
    }
    """
    # Load file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Parse the HTML content
    try:
        soup = BeautifulSoup(content, "html.parser")
        data = {
            "title": (
                soup.title.string
                if soup.title and hasattr(soup.title, "string")
                else "No title found"
            ),
            "headers": [],
        }
        # Extract all headers and paragraphs
        for element in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
            ]
        ):
            content = {
                "tag": element.name,
                "text": element.get_text(strip=False),
                "html": str(element),
            }
            data["headers"].append(content)
    except Exception as e:
        print(f"Error parsing HTML content from {file_path}: {e}")
        return

    # Save extracted data as .json
    json_filename = os.path.basename(file_path).replace(".html", ".json")
    json_folder = os.path.join(RAG_DIR, "crawl_temp/processed_text")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    json_path = os.path.join(json_folder, json_filename)

    try:
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Extracted data saved to {json_path}")
    except Exception as e:
        print(f"Error saving JSON to {json_path}: {e}")


def load_JSON_files(folder_path):
    """
    Load JSON files from a directory and parse them into Document objects.

    Processes each JSON file in the directory by:
    1. Reading and parsing the JSON content
    2. Extracting header texts from the JSON structure
    3. Creating Document objects with the text content and metadata

    :param folder_path: Path to the directory containing JSON files
    :type folder_path: str
    :return: List of Document objects with parsed content and metadata
    :rtype: list[Document]
    :raises OSError: If the folder_path doesn't exist or isn't accessible
    :raises json.JSONDecodeError: If any JSON file is malformed
    :raises Exception: If Document creation fails

    Each Document object contains:
    - page_content: The extracted text from headers
    - metadata: A dictionary with 'source' set to the original filename

    Example:
        >>> docs = load_JSON_files('/path/to/json/files')
        >>> print(docs[0].page_content)
        'Header text content'
        >>> print(docs[0].metadata)
        {'source': 'original.json'}
    """
    JSON_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract header texts from the JSON structure
                    for header in data.get("headers", []):
                        text = header.get("text", "")
                        if text:
                            JSON_files.append(
                                Document(
                                    page_content=text, metadata={"source": filename}
                                )
                            )
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    return JSON_files


def refine_text():
    """
    Execute the process of loading, parsing, and saving HTML content as JSON.

    Main orchestration function that:
    1. Loads HTML files from the html_files directory
    2. Parses each HTML file to extract structured content
    3. Saves the extracted content as JSON files

    The function uses the RAG_DIR configuration to locate input files
    and store output files in the appropriate directories.

    :return: None
    :raises Exception: If the HTML processing pipeline fails at any stage

    Directory structure used:
    - Input: RAG_DIR/crawl_temp/html_files/*.html
    - Output: RAG_DIR/crawl_temp/processed_text/*.json
    """
    # Load HTML files from "html_files" directory
    html_files_dir = os.path.join(RAG_DIR, "crawl_temp/html_files")
    html_files = load_html_files(html_files_dir)

    # Parse and save JSON content for each HTML file individually
    for html_file in html_files:
        parse_and_save(html_file)


# Main Function
def main():
    """
    Main entry point for the HTML refinement process.

    Executes the refine_text() function to process HTML files
    and handles any exceptions that occur during processing.

    :return: None
    :raises SystemExit: If critical errors occur during processing
    """
    refine_text()


if __name__ == "__main__":
    main()
"""
HTML content refinement and processing utilities.

This module provides functionality for processing HTML files and converting them into
a more structured JSON format. It extracts meaningful content like titles, headers,
and paragraphs while preserving the document structure.

The module supports:
- Loading HTML files from a directory
- Parsing HTML content using BeautifulSoup
- Extracting structured content (titles, headers, paragraphs)
- Saving processed content as JSON
- Converting JSON back into Document objects for further processing

The processed documents maintain metadata about their source and structure while
making the content more accessible for NLP tasks.

Note:
    This module requires BeautifulSoup4 for HTML parsing and uses the RAG_DIR
    configuration from indigobot.config.

Example:
    To process a directory of HTML files:
    
    >>> from indigobot.utils.refine_html import refine_text
    >>> refine_text()  # Process all HTML files in the configured directory
"""
