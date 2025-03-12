#!/usr/bin/env python3
"""Convert an imported PDF to a HTML document for parsing."""

import argparse
import logging
import requests
import markdown
import urllib.parse as urlparse
import tempfile
import pymupdf4llm

from pathlib import Path

REQUESTS_TIMEOUT = 10


def download_pdf(url):
    logging.info("Downloading PDF from %r", url)
    try:
        response = requests.get(url, stream=True, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error("Error downloading PDF: %s", e)
        return None

    filename = Path(urlparse.urlparse(url).path).name
    if not filename.endswith(".pdf"):
        filename = "downloaded.pdf"

    temp_pdf_path = Path(tempfile.gettempdir()) / filename
    with open(temp_pdf_path, "wb") as pdf_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_file.write(chunk)
    logging.info("Downloaded PDF to %r", temp_pdf_path)
    return temp_pdf_path


def url_arg_handler(url):
    parsed_url = urlparse.urlparse(url)

    # Not HTTP: interpret as a file path
    if parsed_url.scheme not in ["http", "https"]:
        parsed_path = Path(url).absolute()

        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None

        return parsed_path.as_uri()
    else:
        # Test connection via HEAD request
        logging.info("Testing URL %r with HEAD request", url)
        try:
            requests.head(url, timeout=REQUESTS_TIMEOUT)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.error("Failed to connect to %r", url)
            logging.error("Error message: %s", e)
            return None
        # Download and return the local file path
        downloaded = download_pdf(url)
        if downloaded is None:
            return None
        return downloaded.as_uri()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Input path or URL")
    parser.add_argument("output", help="Output dir")
    args = parser.parse_args()

    pdf_source = url_arg_handler(args.path)
    if pdf_source is None:
        logging.error("Invalid input path or URL")
        exit(1)

    # Convert the file URI to an absolute path
    pdf_path = Path(urlparse.urlparse(pdf_source).path).absolute()
    if not pdf_path.is_file():
        logging.error("File %r not found", pdf_path)
        exit(1)

    md_text = pymupdf4llm.to_markdown(pdf_path)
    html = markdown.markdown(md_text)

    output_path = Path(args.output).joinpath("output.html")
    with open(output_path, "x", encoding="utf-8") as output_file:
        output_file.write(html)


if __name__ == "__main__":
    main()
