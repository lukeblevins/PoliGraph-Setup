#!/usr/bin/env python3
"""PDF (or Web Page) -> HTML Converter Script.

This utility accepts one input argument that can be:
    * A local filesystem path to a PDF (`/path/to/file.pdf`).
    * A `file://` URI pointing at a local PDF.
    * A local filesystem path / `file://` URI to an **HTML/web** resource (we will
        render it to PDF first using Playwright, then extract text).
    * An `http(s)://` URL that is either:
            - A direct link to a PDF (detected via extension), which we download, or
            - A regular web page which we load headlessly and print to PDF.

Once a PDF is obtained (either downloaded or generated via headless Chromium),
we convert its textual content to Markdown via `pymupdf4llm.to_markdown`, then
render that Markdown to HTML and save `output.html` alongside the intermediate
`output.pdf` inside the specified output directory.

This script is intentionally stand‑alone so it can be invoked before the rest
of the PoliGraph pipeline when only a PDF/HTML normalization step is needed.
"""

import argparse
import logging
import sys
import requests
import markdown
import urllib.parse as urlparse
import pymupdf4llm

from pathlib import Path
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

REQUESTS_TIMEOUT = 10  # Seconds for network & page load timeouts


def create_pdf(url, args):
    """Render a web page to a tagged PDF and return its temporary path.

    We use Playwright (Chromium / Edge channel) in headless mode to:
      1. Navigate to the URL.
      2. Wait (best effort) for network idle.
      3. Capture a PDF (`output.pdf`).

    The function performs light error handling:
      * Collects response statuses and fails on any >=400 for navigated frames.
      * On fatal issues, logs and exits the process (mirrors original CLI style).

    Parameters
    ----------
    url : str
        Remote web page URL.
    args : argparse.Namespace
        Parsed arguments containing the output directory path.
    """
    if url is None:
        logging.error("URL failed pre-tests. Exiting...")
        sys.exit(-1)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(bypass_csp=True)
        context.set_default_timeout(REQUESTS_TIMEOUT * 1000)

        def error_cleanup(msg):
            logging.error(msg)
            context.close()
            browser.close()
            sys.exit(-1)

        page = context.new_page()
        page.set_default_timeout(REQUESTS_TIMEOUT * 1000)
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
        page.pdf(path=temp_pdf_path, tagged=True)

        logging.info("Saved to %s", temp_pdf_path)
        context.close()
        browser.close()
        return temp_pdf_path


def download_pdf(url, args):
    """Download a remote PDF and store it as `output.pdf` in the output dir.

    We stream the response to keep memory usage modest and rely on `requests`
    for network + HTTP error handling. The filename in the URL path is not
    preserved (we standardize to `output.pdf`) to simplify downstream code.
    """
    logging.info("Downloading PDF from %r", url)
    try:
        response = requests.get(url, stream=True, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error("Error downloading PDF: %s", e)
        raise e

    filename = Path(urlparse.urlparse(url).path).name
    if not filename.endswith((".pdf", ".PDF")):
        filename = "downloaded.pdf"

    temp_pdf_path = Path(args.output).joinpath("output.pdf")
    # temp_pdf_path = Path(tempfile.gettempdir()) / filename
    with open(temp_pdf_path, "wb") as pdf_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_file.write(chunk)
    logging.info("Downloaded PDF to %r", temp_pdf_path)
    return temp_pdf_path


def url_arg_handler(url, args):
    """Normalize the user-supplied input (local path / URI / URL) into a PDF.

    Decision flow:
      1. If input is a `file://` URI or a plain path: verify existence and
         return its filesystem path.
      2. If remote and ends with `.pdf`: download it.
      3. Otherwise: treat as a web page and print to PDF via Playwright.

    Returns
    -------
    str | None
        Path to a local PDF file, or `None` on unrecoverable error.
    """
    parsed_url = urlparse.urlparse(url)
    logging.info("Parsed URL: %s", parsed_url)
    # Local file via file:// URI
    if parsed_url.scheme == "file":
        parsed_path = Path(parsed_url.path)
        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None
        logging.info("Local file path %r is valid", parsed_path)
        # Return plain filesystem path for downstream libs
        return str(parsed_path)
    # Plain local path (no scheme/netloc)
    if parsed_url.scheme == "" and parsed_url.netloc == "":
        logging.info("Interpreting %r as a local file path", url)
        parsed_path = Path(url).absolute()
        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None
        logging.info("Local file path %r is valid", parsed_path)
        return str(parsed_path)
    # Remote URL (http/https or with netloc)
    if parsed_url.scheme not in ("http", "https"):
        # Default to https if a non-empty netloc exists without scheme
        parsed_url = parsed_url._replace(scheme="https")
        url = parsed_url.geturl()

    # Determine if website or PDF link
    if url.endswith((".pdf", ".PDF")):
        logging.info("Interpreting %r as a PDF URL", url)
        # Download and return the local file path
        downloaded = download_pdf(url, args)
        if downloaded is None:
            logging.error("Failed to download PDF from %r", url)
            return None
        logging.info("PDF downloaded successfully from %r", url)
        return downloaded
    else:
        logging.info("Interpreting %r as a website URL", url)
        exported = create_pdf(url, args)
        if exported is None:
            logging.error("Failed to create PDF from website %r", url)
            return None
        logging.info("PDF created successfully from website %r", url)
        return exported


def main(url, output):
    """Entry point used by both the CLI and potential programmatic calls.

    After producing / locating a PDF, we:
      * Convert PDF -> Markdown (structured extraction via PyMuPDF LLM adapter).
      * Convert Markdown -> HTML using the `markdown` library.
      * Write `output.html` in the provided output directory.
    """
    args = argparse.Namespace(url=url, output=output)
    pdf_path = url_arg_handler(args.url, args)

    # Convert the PDF to Markdown and then to HTML
    md_text = pymupdf4llm.to_markdown(pdf_path)
    html = markdown.markdown(md_text)

    output_path = Path(args.output).joinpath("output.html")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(html)


if __name__ == "__main__":
    # Fallback to original CLI behavior: enforce exactly two positional args.
    if len(sys.argv) != 3:
        print("usage: pdf_parser.py <url_or_path> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
