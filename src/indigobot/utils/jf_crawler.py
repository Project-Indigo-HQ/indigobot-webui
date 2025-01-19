import os
import random
import time
import xml.etree.ElementTree as ET

import urllib.robotparser
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from indigobot.config import RAG_DIR, sitemaps, test_url_list_HTML,test_url_list_XML

USER_AGENT = "SocialServiceChatBot/1.0 (StudentProject; Python)"

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

#TODO remember to choose which one will work better
# Read robots.txt from the website and store its information into "terms"
def fetch_robot_txt_old(base_url):
    # Intialize robot parser and check if allowed
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{base_url}/robots.txt")
    rp.read()
    return rp

def fetch_robot_txt_new(base_url):
    """
    Fetch and parse robots.txt, explicitly handling edge cases for empty Disallow directives.
    
    :param base_url: The base URL of the website (e.g., "https://centralcityconcern.org").
    :return: An updated RobotFileParser instance.
    """
    robots_url = f"{base_url}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()

    # Fetch the robots.txt file
    response = requests.get(robots_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch robots.txt from {robots_url}")

    # Parse the robots.txt content
    robots_txt_content = response.text.strip()
    if "Disallow:" in robots_txt_content and "Disallow:\n" not in robots_txt_content:
        # Handle empty Disallow explicitly
        lines = robots_txt_content.splitlines()
        adjusted_lines = []
        for line in lines:
            if line.strip().lower().startswith("disallow:") and line.strip() == "Disallow:":
                adjusted_lines.append("Disallow: /")  # Fallback if urllib.robotparser misinterprets
            else:
                adjusted_lines.append(line)
        robots_txt_content = "\n".join(adjusted_lines)

    # Feed the (possibly adjusted) content into the RobotFileParser
    rp.parse(robots_txt_content.splitlines())
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
    rp = fetch_robot_txt_new(base_url)

    if not rp.can_fetch(USER_AGENT,url):
        print(f"Disallowed by robots.txt: {url}")
        return None

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
    rp = fetch_robot_txt_new(temp_url)
    
    while urls_to_check:
        time.sleep(1)
        current_url = urls_to_check.pop(0)

        # Check permission
        if not rp.can_fetch(USER_AGENT, current_url):
            print(f"Disallowed by robots.txt: {current_url}")
            continue

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
    Download HTML content from a list of URLs and save it to a file.

    :param urls: List of URLs to download HTML from.
    :type urls: list[str]
    :param session: The requests session to use for downloading.
    :type session: requests.Session
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
    Crawls a website starting from the given sitemaps, downloading HTML content.

    This function orchestrates the crawling process by initiating a session,
    extracting URLs from sitemaps, and downloading the corresponding HTML pages.
    """

    session = start_session()
    url_list = []

    # Scrape URLs from the sitemap
    """
    for page in sitemaps:
        url_list.extend(retrieve_final_urls(page,session))
    """
    #TODO remember to change back
    # Temp
    for page in test_url_list_XML + test_url_list_HTML:
        url_list.extend(retrieve_final_urls(page,session))
        print(f"current page{page}")
    print("crawl finish")
    # Temp

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
