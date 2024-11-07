import requests
import xml.etree.ElementTree as ET
import os
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
    
def extract_xml(xml):
    sitemap = ET.fromstring(xml)
    url_list = []
    for url_element in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):  # Iterate through each <url> tag
        loc = url_element.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text  # Get the <loc> tag value
        url_list.append(loc)
    return url_list

def main():
    # URL of the housing sitemap
    housing_sitemap_url = "https://centralcityconcern.org/housing-sitemap.xml"
    healthcare_sitemap_url = "https://centralcityconcern.org/healthcare-sitemap.xml"
    recovery_sitemap_url = "https://centralcityconcern.org/recovery-sitemap.xml"
    jobs_sitemap_url = "https://centralcityconcern.org/jobs-sitemap.xml"

    # Scrrape Housing URL and then safe to file
    housing_urls = extract_xml(fetch_xml(housing_sitemap_url))
    healthcare_url = extract_xml(fetch_xml(healthcare_sitemap_url))
    recovery_url = extract_xml(fetch_xml(recovery_sitemap_url))
    jobs_url = extract_xml(fetch_xml(jobs_sitemap_url))

    # Output all housing URLs
    print("Extracted Housing URLs:")
    for url in housing_urls + healthcare_url + recovery_url + jobs_url:
        print(url)


    # Ensure 'urls' directory exists
    if not os.path.exists("urls"):
        os.makedirs("urls")
    # Save URLs to a file
    with open("urls/housing_urls.txt", "w") as file:
        for url in housing_urls:
            file.write(url + "\n")
    
    with open("urls/healthcare_url.txt", "w") as file:
        for url in healthcare_url:
            file.write(url + "\n")

    with open("urls/recovery_url.txt", "w") as file:
        for url in recovery_url:
            file.write(url + "\n")

    with open("urls/jobs_url.txt", "w") as file:
        for url in jobs_url:
            file.write(url + "\n")

if __name__ == "__main__":
    main()