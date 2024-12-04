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
    urls_to_cehck = [base_url]
    final_urls = []
    
    while urls_to_cehck:
        time.sleep(1)
        current_url = urls_to_cehck.pop(0)
        # If reached a stiemap, extract the URLs and add them to the list to check
        if is_sitemap(current_url,session):
            print(f"Processing sitemap: {current_url}")
            sitemaps_content = fetch_xml(current_url, session)
            extracted_urls = extract_xml(sitemaps_content)
            urls_to_cehck.extend(extracted_urls)
        # If it is not a sitemap, add it to the final_urls
        else:
            print(f"Found terminal URL: {current_url}")
            final_urls.append(current_url)
    return final_urls


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
