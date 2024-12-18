import os
import random
import time
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from indigobot.config import RAG_DIR, sitemaps


# create a REST session
def start_session():
    """
    Create and configure a REST session with retry mechanisms.

    :return: A configured requests session.
    :rtype: requests.Session
    """
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# Function to fetch and parse XML from a URL with retry mechanism
def fetch_xml(url, session):
    """
    Fetch XML content from a given URL using a session with retries.

    :param url: The URL to fetch XML from.
    :type url: str
    :param session: The requests session to use for fetching.
    :type session: requests.Session
    :return: The XML content of the response.
    :rtype: bytes
    :raises Exception: If the URL cannot be fetched successfully.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = session.get(url, headers=headers)
    if response.status_code == 200:
        time.sleep(5)
        return response.content
    else:
        raise Exception(
            f"Failed to fetch XML from {url}, Status Code: {response.status_code}"
        )


# Functionn to parse a list of url to resource page from sitemap
def extract_xml(xml):
    """
    Parse XML content to extract URLs.

    :param xml: The XML content to parse.
    :type xml: bytes
    :return: A list of URLs extracted from the XML.
    :rtype: list[str]
    """
    sitemap = ET.fromstring(xml)
    url_list = []
    for url_element in sitemap.findall(
        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"
    ):  # Iterate through each <url> tag
        loc = url_element.find(
            "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
        ).text  # Get the <loc> tag value
        url_list.append(loc)
    return url_list


# Load each of the .txt file under "urls/"
def load_urls(folder_path):
    """
    Load URLs from text files in a specified folder.

    :param folder_path: Path to the folder containing URL text files.
    :type folder_path: str
    :return: A list of URLs read from the files.
    :rtype: list[str]
    """
    urls = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                urls.extend(file.read().splitlines())
    return urls


# Get the HTML file for each URL and save it
def download_and_save_html(urls, session):
    """
    Download HTML content from a list of URLs and save it to a file.

    :param urls: List of URLs to download HTML from.
    :type urls: list[str]
    :param session: The requests session to use for downloading.
    :type session: requests.Session
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        time.sleep(random.randint(3, 6))
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            # Extract last section of url as file name
            filename = url.rstrip("/").split("/")[-1] + ".html"

            html_files_dir = os.path.join(RAG_DIR, "crawl_temp", "html_files")
            os.makedirs(html_files_dir, exist_ok=True)

            # save the content to html
            with open(
                os.path.join(html_files_dir, filename), "w", encoding="utf-8"
            ) as file:
                file.write(response.text)
                print(f"Save html content to {html_files_dir}")
        else:
            print(f"Faile to fetch {url}, Status code: {response.status_code}")


def parse_url_and_save(sitemap_url, target_file_name, session):
    """
    Parse URLs from a sitemap and save them to a file.

    :param sitemap_url: The URL of the sitemap to parse.
    :type sitemap_url: str
    :param target_file_name: The name of the file to save URLs.
    :type target_file_name: str
    :param session: The requests session to use for fetching.
    :type session: requests.Session
    """
    urls = extract_xml(fetch_xml(sitemap_url, session))

    # Output all housing URLs
    print("Extracted Housing URLs:")
    for url in urls:
        print(url)

    # Ensure 'urls' directory exists
    if not os.path.exists(os.path.join(RAG_DIR, "crawl_temp/extracted_urls")):
        os.makedirs("crawl_temp/extracted_urls")

    # Save URLs to a file
    with open(f"crawl_temp/extracted_urls/{target_file_name}.txt", "w") as file:
        for url in urls:
            file.write(url + "\n")
    time.sleep(5)


def parse_url(sitemap_url, session):
    """
    Parse URLs from a sitemap without saving them to a file.

    :param sitemap_url: The URL of the sitemap to parse.
    :type sitemap_url: str
    :param session: The requests session to use for fetching.
    :type session: requests.Session
    :return: A list of URLs extracted from the sitemap.
    :rtype: list[str]
    """
    urls = []
    page_content = extract_xml(fetch_xml(sitemap_url, session))

    # Display all  URLs
    print("Extracting URLs:")
    for url in page_content:
        print(url)
        urls.append(url)

    return urls


def crawl():
    """
    Crawls a website starting from the given sitemaps, downloading HTML content.

    This function orchestrates the crawling process by initiating a session,
    extracting URLs from sitemaps, and downloading the corresponding HTML pages.
    """

    session = start_session()
    url_list = []

    # Scrape URLs from the sitemap
    for page in sitemaps:
        url_list.extend(parse_url(page, session))

    # Download all resource page as html
    download_and_save_html(url_list, session)

    print("\nThe crawler is finished")


def main():
    """
    The main entry point for the crawler script.
    """
    crawl()


if __name__ == "__main__":
    main()
"""
A robust web crawler for extracting content from sitemaps and web pages.

This module implements a web crawler with built-in retry mechanisms and polite
crawling behavior. It can process XML sitemaps, extract URLs, and download HTML
content while respecting rate limits and server constraints.

Features:
- Configurable retry mechanism for failed requests
- Random delays between requests to avoid overwhelming servers
- XML sitemap parsing and URL extraction
- Bulk HTML content downloading
- File-based URL storage and management
- Session management with custom headers

The crawler implements best practices for web scraping including:
- User-agent identification
- Rate limiting
- Error handling
- Retry logic
- Content verification

Note:
    Configuration settings including sitemaps and RAG_DIR are pulled from
    indigobot.config.
"""
