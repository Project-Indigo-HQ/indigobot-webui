# some library that maybe useful

import bs4 # can be use forparsing HTML and XML
import selenium # can be use to interact with web pages
import scrapy # can be use to scrap content from web
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

# Main function
def main():
    # Load urls from all .txt files
    urls = load_urls("./urls")

    download_and_save_html(urls)

if __name__ == "__main__":
    main()
