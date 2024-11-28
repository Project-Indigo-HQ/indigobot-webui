import json
import os
import unittest
from unittest.mock import mock_open, patch

from langchain.schema import Document

from indigobot.utils.refine_html import (
    load_html_files,
    load_JSON_files,
    parse_and_save,
    refine_text,
)


class TestRefineHtml(unittest.TestCase):
    def setUp(self):
        self.test_html = """
        <html>
            <head>
                <title>Test Page</title>
            </head>
            <body>
                <h1>Main Header</h1>
                <h2>Sub Header</h2>
                <div>
                    <h3>Section Header</h3>
                    <p>Some content</p>
                </div>
            </body>
        </html>
        """
        self.test_json = {
            "title": "Test Page",
            "headers": [
                {"tag": "h1", "text": "Main Header", "html": "<h1>Main Header</h1>"},
                {"tag": "h2", "text": "Sub Header", "html": "<h2>Sub Header</h2>"},
            ],
        }

    def test_load_html_files(self):
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1.html", "test2.html", "other.txt"]
            files = load_html_files("/fake/path")
            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.endswith(".html") for f in files))

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="<html><title>Test</title></html>",
    )
    def test_parse_and_save_success(self, mock_file):
        with patch("os.path.exists") as mock_exists, patch(
            "os.makedirs"
        ) as mock_makedirs, patch("json.dump") as mock_json_dump:

            mock_exists.return_value = False
            parse_and_save("test.html")
            mock_makedirs.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_parse_and_save_file_not_found(self):
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = FileNotFoundError()
            parse_and_save("nonexistent.html")
            # Should handle the error gracefully without raising exception

    def test_parse_and_save_with_real_html(self):
        m = mock_open(read_data=self.test_html)
        with patch("builtins.open", m), patch("os.path.exists") as mock_exists, patch(
            "os.makedirs"
        ) as mock_makedirs, patch("json.dump") as mock_json_dump:

            mock_exists.return_value = False
            parse_and_save("test.html")

            # Verify JSON structure
            calls = mock_json_dump.call_args_list
            self.assertEqual(len(calls), 1)
            saved_data = calls[0][0][0]  # First arg of first call
            self.assertEqual(saved_data["title"], "Test Page")
            self.assertEqual(len(saved_data["headers"]), 3)  # h1, h2, h3

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"headers": [{"text": "Test Header"}]}',
    )
    @patch("os.listdir")
    def test_load_JSON_files(self, mock_listdir, mock_file):
        mock_listdir.return_value = ["test1.json", "test2.json", "other.txt"]

        documents = load_JSON_files("/fake/path")
        self.assertEqual(len(documents), 2)  # Two JSON files processed
        self.assertEqual(documents[0].page_content, "Test Header")
        self.assertEqual(documents[0].metadata["source"], "test1.json")
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        self.assertEqual(len(documents), 2)  # One doc per JSON file

    @patch("indigobot.utils.jf_crawler.refine_html.load_html_files")
    @patch("indigobot.utils.jf_crawler.refine_html.parse_and_save")
    def test_refine_text(self, mock_parse_save, mock_load_files):
        mock_load_files.return_value = ["test1.html", "test2.html"]
        refine_text()
        self.assertEqual(mock_parse_save.call_count, 2)
        mock_load_files.assert_called_once()


if __name__ == "__main__":
    unittest.main()
