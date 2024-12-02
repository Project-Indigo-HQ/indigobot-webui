import os
import unittest
from unittest.mock import Mock, mock_open, patch

from indigobot.utils.jf_crawler import (
    download_and_save_html,
    extract_xml,
    fetch_xml,
    load_urls,
    parse_url,
    start_session,
)


class TestCrawler(unittest.TestCase):
    def setUp(self):
        self.test_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
            </url>
        </urlset>"""

    def test_start_session(self):
        session = start_session()
        self.assertIsNotNone(session)
        self.assertEqual(session.adapters["https://"].max_retries.total, 5)

    @patch("requests.Session")
    def test_fetch_xml_success(self, mock_session):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<xml>test</xml>"
        mock_session.return_value.get.return_value = mock_response

        session = mock_session()
        result = fetch_xml("https://example.com", session)
        self.assertEqual(result, b"<xml>test</xml>")

    @patch("requests.Session")
    def test_fetch_xml_failure(self, mock_session):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response

        session = mock_session()
        with self.assertRaises(Exception):
            fetch_xml("https://example.com", session)

    def test_extract_xml(self):
        urls = extract_xml(self.test_xml.encode())
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], "https://example.com/page1")
        self.assertEqual(urls[1], "https://example.com/page2")

    def test_load_urls(self):
        mock_data = {
            'test_urls/test1.txt': 'https://example.com/test1\n',
            'test_urls/test2.txt': 'https://example.com/test2\n'
        }
        
        def mock_open_func(filename, *args, **kwargs):
            m = mock_open(read_data=mock_data[filename])
            return m(*args, **kwargs)
            
        with patch('builtins.open', mock_open_func), \
             patch("os.path.exists") as mock_exists, \
             patch("os.listdir") as mock_listdir:
            
            mock_exists.return_value = True
            mock_listdir.return_value = ["test1.txt", "test2.txt"]
            urls = load_urls("test_urls")
            self.assertEqual(len(urls), 2)
            self.assertEqual(urls[0], "https://example.com/test1")
            self.assertEqual(urls[1], "https://example.com/test2")

    @patch("requests.Session")
    def test_download_and_save_html(self, mock_session):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html>test</html>"
        mock_session.return_value.get.return_value = mock_response

        test_urls = ["https://example.com/page1"]
        session = mock_session()

        with patch("os.makedirs") as mock_makedirs, patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            download_and_save_html(test_urls, session)
            # Verify makedirs was called
            mock_makedirs.assert_called_once()
            # Verify file was opened for writing
            mock_file.assert_called_once()

    @patch("indigobot.utils.jf_crawler.fetch_xml")
    @patch("indigobot.utils.jf_crawler.extract_xml")
    def test_parse_url(self, mock_extract_xml, mock_fetch_xml):
        mock_extract_xml.return_value = ["https://example.com/page1"]
        mock_fetch_xml.return_value = self.test_xml.encode()

        session = Mock()
        urls = parse_url("https://example.com/sitemap.xml", session)

        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0], "https://example.com/page1")


if __name__ == "__main__":
    unittest.main()
