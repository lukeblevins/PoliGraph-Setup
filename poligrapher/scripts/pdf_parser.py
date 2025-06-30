#!/usr/bin/env python3
"""Convert an imported PDF to a HTML document for parsing."""

import argparse
import logging
import sys
import requests
import markdown
import urllib.parse as urlparse
import pymupdf4llm

from pathlib import Path
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

REQUESTS_TIMEOUT = 10


async def create_pdf(url, args):
    if url is None:
        logging.error("URL failed pre-tests. Exiting...")
        sys.exit(-1)

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        context = await browser.new_context(bypass_csp=True)

        async def error_cleanup(msg):
            logging.error(msg)
            await context.close()
            await browser.close()
            sys.exit(-1)

        page = await context.new_page()
        await page.set_viewport_size({"width": 1080, "height": 1920})
        logging.info("Navigating to %r", url)

        # Record HTTP status and navigated URLs so we can check errors later
        url_status = dict()
        navigated_urls = []
        page.on("response", lambda r: url_status.update({r.url: r.status}))
        page.on(
            "framenavigated",
            lambda f: f.parent_frame is None and navigated_urls.append(f.url),
        )

        await page.emulate_media(media="print")
        await page.goto(url)

        try:
            await page.wait_for_load_state("networkidle")
        except PlaywrightTimeoutError:
            logging.warning("Cannot reach networkidle but will continue")

        # Check HTTP errors
        for url in navigated_urls:
            if (status_code := url_status.get(url, 0)) >= 400:
                await error_cleanup(f"Got HTTP error {status_code}")

        output_dir = Path(args.output)
        temp_pdf_path = output_dir / "output.pdf"
        await page.pdf(path=temp_pdf_path, tagged=True)

        logging.info("Saved to %s", temp_pdf_path)
        await context.close()
        await browser.close()
        return temp_pdf_path


def download_pdf(url, args):
    logging.info("Downloading PDF from %r", url)
    try:
        response = requests.get(url, stream=True, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error("Error downloading PDF: %s", e)
        return None

    filename = Path(urlparse.urlparse(url).path).name
    if not filename.endswith([".pdf", ".PDF"]):
        filename = "downloaded.pdf"

    temp_pdf_path = Path(args.output).joinpath("output.pdf")
    # temp_pdf_path = Path(tempfile.gettempdir()) / filename
    with open(temp_pdf_path, "wb") as pdf_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_file.write(chunk)
    logging.info("Downloaded PDF to %r", temp_pdf_path)
    return temp_pdf_path


async def url_arg_handler(url, args):
    parsed_url = urlparse.urlparse(url)
    logging.info("Parsed URL: %s", parsed_url)
    # check if URL or local file:
    if not parsed_url.scheme in ["https", "http"] and not parsed_url.netloc:
        # No scheme or netloc: local file path
        logging.info("Interpreting %r as a local file path", url)
        parsed_path = Path(url).absolute()

        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None

        logging.info("Local file path %r is valid", parsed_path)
        return parsed_path.as_uri()
    else:
        # No scheme: assume HTTPS
        parsed_url = parsed_url._replace(scheme="https")
        url = parsed_url.geturl()

        # Determine if website or PDF link
        if url.endswith(".pdf") or url.endswith(".PDF"):
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
            exported = await create_pdf(url, args)
            if exported is None:
                logging.error("Failed to create PDF from website %r", url)
                return None
            logging.info("PDF created successfully from website %r", url)
            return exported


async def main(url, output):
    args = argparse.Namespace(url=url, output=output)
    pdf_path = await url_arg_handler(args.url, args)

    # Convert the PDF to Markdown and then to HTML
    md_text = pymupdf4llm.to_markdown(pdf_path)
    html = markdown.markdown(md_text)

    output_path = Path(args.output).joinpath("output.html")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(html)


if __name__ == "__main__":
    # fallback to original CLI behavior
    if len(sys.argv) != 3:
        print("usage: pdf_parser.py <url_or_path> <output_dir>")
        sys.exit(1)

    import asyncio

    asyncio.run(main(sys.argv[1], sys.argv[2]))
