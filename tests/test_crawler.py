import os
import unittest
from unittest.mock import Mock, mock_open, patch

from indigobot.utils.jf_crawler import (
    download_and_save_html,
    extract_xml,
    fetch_xml,
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



if __name__ == "__main__":
    unittest.main()
