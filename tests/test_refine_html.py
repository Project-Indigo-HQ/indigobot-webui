import json
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
        with patch("os.listdir") as mock_listdir, patch(
            "os.path.isfile"
        ) as mock_isfile:
            mock_listdir.return_value = ["test1.html", "test2.html", "other.txt"]
            mock_isfile.return_value = True
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
            self.assertEqual(len(saved_data["headers"]), 4)  # h1, h2, h3, p
            # Verify content
            headers = saved_data["headers"]
            self.assertEqual(headers[0]["tag"], "h1")
            self.assertEqual(headers[0]["text"], "Main Header")
            self.assertEqual(headers[3]["tag"], "p")
            self.assertEqual(headers[3]["text"], "Some content")

    @patch("os.listdir")
    def test_load_JSON_files(self, mock_listdir):
        mock_data = {"headers": [{"text": "Test Header 1"}, {"text": "Test Header 2"}]}
        m = mock_open(read_data=json.dumps(mock_data))
        mock_listdir.return_value = ["test1.json", "test2.json", "other.txt"]

        with patch("builtins.open", m):
            documents = load_JSON_files("/fake/path")
            self.assertEqual(len(documents), 4)  # 2 headers × 2 files
            self.assertEqual(documents[0].page_content, "Test Header 1")
            self.assertEqual(documents[0].metadata["source"], "test1.json")
            self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_load_JSON_files_invalid_json(self):
        with patch("os.listdir") as mock_listdir, patch(
            "builtins.open", mock_open(read_data="invalid json")
        ):
            mock_listdir.return_value = ["test1.json"]
            documents = load_JSON_files("/fake/path")
            self.assertEqual(len(documents), 0)  # Should handle invalid JSON gracefully

    @patch("indigobot.utils.refine_html.load_html_files")
    @patch("indigobot.utils.refine_html.parse_and_save")
    def test_refine_text(self, mock_parse_save, mock_load_files):
        mock_load_files.return_value = ["test1.html", "test2.html"]
        refine_text()
        self.assertEqual(mock_parse_save.call_count, 2)
        mock_load_files.assert_called_once()


if __name__ == "__main__":
    unittest.main()
