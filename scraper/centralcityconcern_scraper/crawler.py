

import requests
import xml.etree.ElementTree as ET
import os
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Function to fetch and parse XML from a URL with retry mechanism
def fetch_xml(url):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = session.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch XML from {url}, Status Code: {response.status_code}")
    

#Functionn to parse a list of url to resource page from sitemap
def extract_xml(xml):
    sitemap = ET.fromstring(xml)
    url_list = []
    for url_element in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):  # Iterate through each <url> tag
        loc = url_element.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text  # Get the <loc> tag value
        url_list.append(loc)
    return url_list

# Load each of the .txt file under "urls/"
def load_urls(folder_path):
    urls = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding = 'utf-8') as file:
                urls.extend(file.read().splitlines())
    return urls

# Get the HTML file for each URL and save it
def download_and_save_html(urls):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    
    for url in urls:
        time.sleep(5)
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            # Extract last section of url as file name
            filename = url.rstrip('/').split('/')[-1] + ".html"

            if not os.path.exists("html_files"):
                os.makedirs("html_files")
            
            # save the content to html
            file_path = os.path.join("html_files", filename)
            with open(file_path, "w", encoding = "utf-8") as file:
                file.write(response.text)
                print(f"Save html content to {file_path}")
        else:
            print(f"Faile to fetch {url}, Status code: {response.status_code}")

def parse_url(sitemap_url,target_file_name):
    urls = extract_xml(fetch_xml(sitemap_url))

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

def main():

    # URL of the housing sitemap
    housing_sitemap_url = "https://centralcityconcern.org/housing-sitemap.xml"
    
    healthcare_sitemap_url = "https://centralcityconcern.org/healthcare-sitemap.xml"
    
    recovery_sitemap_url = "https://centralcityconcern.org/recovery-sitemap.xml"
    
    jobs_sitemap_url = "https://centralcityconcern.org/jobs-sitemap.xml"

    # Scrrape Housing URL and then safe to file
    parse_url(housing_sitemap_url, "housing_urls")
    parse_url(healthcare_sitemap_url, "healthcare_url")
    parse_url(recovery_sitemap_url, "recovery_url")
    parse_url(jobs_sitemap_url, "jobs_url")

    # Load urls from all .txt files
    urls = load_urls("./urls")
    
    # Download all resource page as html
    download_and_save_html(urls)

if __name__ == "__main__":
    main()