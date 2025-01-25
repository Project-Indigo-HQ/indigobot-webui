import os
import random
import time
import xml.etree.ElementTree as ET

import urllib.robotparser
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from time import sleep

from indigobot.config import RAG_DIR, sitemaps, url_list_XML

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SocialServiceChatBot/1.0 (Python)"

# create a REST session
def start_session():
    """
    Create and configure a REST session with retry mechanisms and backoff.

    Creates a requests Session object configured with:
    - Up to 5 retries for failed requests
    - Exponential backoff starting at 1 second
    - Retries on specific HTTP error codes (403, 500, 502, 503, 504)

    :return: A configured requests Session object ready for making HTTP requests
    :rtype: requests.Session
    :raises ImportError: If the requests package is not available
    """
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_robot_txt(base_url):
    """
    Fetch and parse robots.txt with a retry mechanism for handling failures.
    
    :param base_url: The base URL of the website (e.g., "https://centralcityconcern.org").
    :param retries: Number of retry attempts for fetching robots.txt (default: 3).
    :param backoff_factor: Multiplier for the delay between retries (default: 2).
    :return: An updated RobotFileParser instance.
    """
    robots_url = f"{base_url}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    retries=3
    backoff_factor=2
    attempt = 0
    while attempt < retries:
        try:
            # Fetch the robots.txt file
            response = requests.get(robots_url, timeout=5)
            if response.status_code == 200:
                robots_txt_content = response.text.strip()
                
                # Handle edge cases for empty Disallow directives
                if "Disallow:" in robots_txt_content and "Disallow:\n" not in robots_txt_content:
                    lines = robots_txt_content.splitlines()
                    adjusted_lines = []
                    for line in lines:
                        if line.strip().lower().startswith("disallow:") and line.strip() == "Disallow:":
                            adjusted_lines.append("Disallow: /")
                        else:
                            adjusted_lines.append(line)
                    robots_txt_content = "\n".join(adjusted_lines)

                # Parse the (possibly adjusted) content into the RobotFileParser
                rp.parse(robots_txt_content.splitlines())
                return rp
            else:
                print(f"Failed to fetch {robots_url} (status code {response.status_code}), retrying...")
        except requests.RequestException as e:
            print(f"Error fetching robots.txt: {e}, retrying...")

        # Retry with exponential backoff
        attempt += 1
        sleep(backoff_factor ** attempt)

    # If all retries fail, configure rp to disallow all
    print(f"Failed to fetch robots.txt from {robots_url} after {retries} retries. Returning a disallow-all policy.")
    rp.parse(["User-agent: *", "Disallow: /"])
    return rp


# Extract the base URL from any URL
def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

# Fetch and parse XML from a URL with retry mechanism
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
    # Permission check
    base_url = get_base_url(url)

    headers = {
        "User-Agent": USER_AGENT
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
    Parse XML content from a sitemap to extract URLs.

    Parses XML using ElementTree and extracts all URLs from <loc> tags
    within <url> elements in the sitemap namespace.

    :param xml: Raw XML content from a sitemap
    :type xml: bytes
    :return: List of URLs found in the sitemap
    :rtype: list[str]
    :raises xml.etree.ElementTree.ParseError: If XML parsing fails
    :raises AttributeError: If expected XML elements are not found
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

# Check if the given URL is a sitemap
def is_sitemap(url, session):
    try:
        content = fetch_xml(url, session)
        root = ET.fromstring(content)
        return root.tag.endswith("sitemapindex") or root.tag.endswith("urlset")
    except Exception:
        return False

# Recursive function to retrive final list of URLs
def retrieve_final_urls(base_url, session):
    """
    Locate content page under to current sitemap

    :param base_url: The starting point, must be a sitemap
    :type base_rul: str
    :param session : The current session 
    :type base_rul: requests.Session
    :return: A list of URLs extracted from the XML.
    :rtype: list[str]
    """
    urls_to_check = [base_url]
    final_urls = []
    temp_url = get_base_url(base_url)
    
    while urls_to_check:
        time.sleep(1)
        current_url = urls_to_check.pop(0)

        # If reached a stiemap, extract the URLs and add them to the list to check
        if is_sitemap(current_url,session):
            print(f"Processing sitemap: {current_url}")
            sitemaps_content = fetch_xml(current_url, session)
            extracted_urls = extract_xml(sitemaps_content)
            urls_to_check.extend(extracted_urls)
        # If it is not a sitemap, add it to the final_urls
        else:
            print(f"Found terminal URL: {current_url}")
            final_urls.append(current_url)

    return final_urls


# Get the HTML file for each URL and save it
def download_and_save_html(urls, session):
    """
    Download HTML content from URLs and save to files.

    For each URL:
    1. Makes GET request with random delay (3-6 seconds)
    2. Checks response status
    3. Extracts filename from URL
    4. Saves HTML content to file in crawl_temp/html_files/

    :param urls: List of URLs to download HTML from
    :type urls: list[str]
    :param session: Requests session for making HTTP requests
    :type session: requests.Session
    :raises OSError: If directory creation or file writing fails
    :raises requests.exceptions.RequestException: If downloads fail
    """
    headers = {
        "User-Agent": USER_AGENT
    }

    for url in urls:
        time.sleep(random.randint(1, 2))
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




def crawl():
    """
    Orchestrate the complete website crawling process.

    Main crawling function that:
    1. Creates and configures a requests Session
    2. Iterates through configured sitemaps from config
    3. Extracts URLs from each sitemap
    4. Downloads HTML content for all extracted URLs
    5. Saves HTML files to crawl_temp directory

    The function implements polite crawling practices:
    - Uses retry mechanisms with backoff
    - Adds delays between requests
    - Uses proper User-Agent identification
    - Handles errors gracefully

    :raises Exception: If critical crawling operations fail
    :raises OSError: If file operations fail
    :raises requests.exceptions.RequestException: If HTTP requests fail
    """

    session = start_session()
    url_list = []

    # Scrape URLs from the sitemap

    for page in url_list_XML + sitemaps:
        url_list.extend(retrieve_final_urls(page,session))
    
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
