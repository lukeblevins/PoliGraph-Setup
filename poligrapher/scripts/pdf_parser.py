#!/usr/bin/env python3
"""Convert an imported PDF to a HTML document for parsing."""

import argparse
import logging
import sys
import requests
import markdown
import urllib.parse as urlparse
import tempfile
import pymupdf4llm

from pathlib import Path
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

REQUESTS_TIMEOUT = 10


def create_pdf(url, args):
    if url is None:
        logging.error("URL failed pre-tests. Exiting...")
        sys.exit(-1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(bypass_csp=True)

        def error_cleanup(msg):
            logging.error(msg)
            context.close()
            browser.close()
            sys.exit(-1)

        page = context.new_page()
        page.set_viewport_size({"width": 1080, "height": 1920})
        logging.info("Navigating to %r", url)

        # Record HTTP status and navigated URLs so we can check errors later
        url_status = dict()
        navigated_urls = []
        page.on("response", lambda r: url_status.update({r.url: r.status}))
        page.on(
            "framenavigated",
            lambda f: f.parent_frame is None and navigated_urls.append(f.url),
        )

        page.emulate_media(media="print")
        page.goto(url)

        try:
            page.wait_for_load_state("networkidle")
        except PlaywrightTimeoutError:
            logging.warning("Cannot reach networkidle but will continue")

        # Check HTTP errors
        for url in navigated_urls:
            if (status_code := url_status.get(url, 0)) >= 400:
                error_cleanup(f"Got HTTP error {status_code}")

        output_dir = Path(args.output)
        temp_pdf_path = output_dir / "output.pdf"
        page.pdf(path=temp_pdf_path)

        logging.info("Saved to %s", temp_pdf_path)
        context.close()
        browser.close()
        return temp_pdf_path


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


def url_arg_handler(url, args):
    parsed_url = urlparse.urlparse(url)

    # check if URL or local file:
    if not parsed_url.scheme and not parsed_url.netloc:
        # No scheme or netloc: local file path
        logging.info("Interpreting %r as a local file path", url)
        parsed_path = Path(url).absolute()

        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None

        return parsed_path.as_uri()
    elif not parsed_url.scheme:
        # No scheme: assume HTTP
        parsed_url = parsed_url._replace(scheme="http")
        url = parsed_url.geturl()
        # Test connection via HEAD request
        logging.info("Testing URL %r with HEAD request", url)
        try:
            requests.head(url, timeout=REQUESTS_TIMEOUT)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.error("Failed to connect to %r", url)
            logging.error("Error message: %s", e)
            return None
        # Determine if website or PDF link
        if url.endswith(".pdf"):
            logging.info("Interpreting %r as a PDF URL", url)
            # Download and return the local file path
            downloaded = download_pdf(url)
            if downloaded is None:
                return None
            return downloaded.as_uri()
        else:
            logging.info("Interpreting %r as a website URL", url)
            exported = create_pdf(url, args)
            return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Input URL or path")
    parser.add_argument("output", help="Output dir")
    args = parser.parse_args()

    pdf_source = url_arg_handler(args.url, args)
    if pdf_source is None:
        logging.error("Invalid input path or URL")
        exit(1)

    # Convert the file URI to an absolute path
    pdf_path = Path(urlparse.urlparse(pdf_source).path).absolute()
    if not pdf_path.is_file():
        logging.error("File %r not found", pdf_path)
        exit(1)

    # Convert the PDF to Markdown and then to HTML
    md_text = pymupdf4llm.to_markdown(pdf_path)
    html = markdown.markdown(md_text)

    output_path = Path(args.output).joinpath("output.html")
    with open(output_path, "x", encoding="utf-8") as output_file:
        output_file.write(html)


if __name__ == "__main__":
    main()
