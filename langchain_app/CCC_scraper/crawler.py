import os
import random
import time
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# create a REST session
def start_sessionn():
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# Function to fetch and parse XML from a URL with retry mechanism
def fetch_xml(url, session):
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
    urls = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                urls.extend(file.read().splitlines())
    return urls


# Get the HTML file for each URL and save it
def download_and_save_html(urls, session):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        time.sleep(random.randint(3, 6))
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            # Extract last section of url as file name
            filename = url.rstrip("/").split("/")[-1] + ".html"

            script_dir = os.path.dirname(os.path.abspath(__file__))
            html_files_dir = os.path.join(script_dir, "html_files")
            if not os.path.exists(html_files_dir):
                os.makedirs(html_files_dir)

            # save the content to html

            with open(html_files_dir, "w", encoding="utf-8") as file:
                file.write(response.text)
                print(f"Save html content to {html_files_dir}")
        else:
            print(f"Faile to fetch {url}, Status code: {response.status_code}")


def parse_url_and_save(sitemap_url, target_file_name, session):

    urls = extract_xml(fetch_xml(sitemap_url, session))

    # Output all housing URLs
    print("Extracted Housing URLs:")
    for url in urls:
        print(url)

    # Ensure 'urls' directory exists
    if not os.path.exists("urls"):
        os.makedirs("urls")

    # Save URLs to a file
    with open(f"urls/{target_file_name}.txt", "w") as file:
        for url in urls:
            file.write(url + "\n")
    time.sleep(5)


def parse_url(sitemap_url, session):
    urls = []
    page_content = extract_xml(fetch_xml(sitemap_url, session))

    # Display all  URLs
    print("Extracting URLs:")
    for url in page_content:
        print(url)
        urls.append(url)

    return urls


# TODO make this work anywhere
def crawl():
    session = start_sessionn()
    url_list = []

    # URL of the sitemap
    sitemaps = [
        "https://centralcityconcern.org/housing-sitemap.xml",
        "https://centralcityconcern.org/healthcare-sitemap.xml",
        "https://centralcityconcern.org/recovery-sitemap.xml",
        "https://centralcityconcern.org/jobs-sitemap.xml",
    ]

    # Scrrape URLs from the sitemap
    for page in sitemaps:
        url_list.extend(parse_url(page, session))

    # Download all resource page as html
    download_and_save_html(url_list, session)

    print("\nThe crawler is finished")


def main():
    crawl()


if __name__ == "__main__":
    main()
