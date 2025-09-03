#!/usr/bin/env python3
"""Download a web page and export the accessibility tree for parsing."""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
import re
import urllib.parse as urlparse

import bs4
import langdetect
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)
import requests
from requests_cache import CachedSession

READABILITY_JS_COMMIT = "04fd32f72b448c12b02ba6c40928b67e510bac49"
READABILITY_JS_URL = (
    f"https://raw.githubusercontent.com/mozilla/readability/{READABILITY_JS_COMMIT}"
)
_READABILITY_JS_CACHE: str | None = None
REQUESTS_TIMEOUT = 10


def get_readability_js():
    """Fetch and cache the Readability.js sources.

    The original code fetched the main script twice; this removes the duplication and
    stores the concatenated JS in a module-level cache to avoid repeated network
    requests when crawling many documents in one process.
    """
    global _READABILITY_JS_CACHE
    if _READABILITY_JS_CACHE is not None:
        return _READABILITY_JS_CACHE

    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    parts: list[str] = []

    for suffix in ("Readability.js", "Readability-readerable.js"):
        res = session.get(f"{READABILITY_JS_URL}/{suffix}", timeout=REQUESTS_TIMEOUT)
        res.raise_for_status()
        parts.append(res.text)

    _READABILITY_JS_CACHE = "\n".join(parts)
    return _READABILITY_JS_CACHE


def url_arg_handler(url):
    parsed_url = urlparse.urlparse(url)

    # Not HTTP(s): interpret as a file path
    if parsed_url.scheme not in ["http", "https"]:
        parsed_path = Path(url).absolute()

        if not parsed_path.is_file():
            raise FileNotFoundError(f"File {url} not found")

        return parsed_path.as_uri()

    # Handle Google Docs URLs
    if (
        parsed_url.hostname == "docs.google.com"
        and not parsed_url.path.endswith("/pub")
        and (
            m := re.match(
                r"/document/d/(1[a-zA-Z0-9_-]{42}[AEIMQUYcgkosw048])", parsed_url.path
            )
        )
    ):
        logging.info("Exporting HTML from Google Docs URL...")

        export_url = f"https://docs.google.com/feeds/download/documents/export/Export?id={m[1]}&exportFormat=html"

        req = requests.get(export_url, timeout=REQUESTS_TIMEOUT)
        req.raise_for_status()

        base64_url = "data:text/html;base64," + base64.b64encode(req.content).decode()
        req.close()
        return base64_url

    return url


def main(url, output, fast: bool = False):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )
    # fast mode can be enabled via parameter or env var POLIGRAPH_FAST=1
    fast = fast or os.environ.get("POLIGRAPH_FAST", "0") in {"1", "true", "True"}

    args = argparse.Namespace(
        url=url, output=output, no_readability_js=False, fast=fast
    )
    access_url = url_arg_handler(args.url)

    firefox_configs = {
        # Bypass CSP so we can always inject scripts
        "security.csp.enable": False,
        # Allow insecure TLS versions
        "security.tls.version.min": 1,
        "security.tls.version.enable-deprecated": True,
        # Prevent some background traffic
        "dom.serviceWorkers.enabled": False,
        "network.websocket.max-connections": 0,
        "media.autoplay.default": 5,
        "media.peerconnection.enabled": False,
        "privacy.trackingprotection.enabled": True,
        "privacy.trackingprotection.lower_network_priority": True,
        "privacy.trackingprotection.socialtracking.enabled": True,
    }

    with sync_playwright() as p:
        t0 = time.monotonic()
        # Firefox generates simpler accessibility tree than chromium
        # Tested on Debian's firefox-esr 91.5.0esr-1~deb11u1
        browser = p.firefox.launch(firefox_user_prefs=firefox_configs, headless=True)
        context = browser.new_context(bypass_csp=True)
        context.set_default_timeout(REQUESTS_TIMEOUT * 1000)

        def error_cleanup(msg):
            logging.error(msg)
            try:
                context.close()
                browser.close()
            finally:
                raise RuntimeError(f"html_crawler failure: {msg}")

        page = context.new_page()
        page.set_default_timeout(REQUESTS_TIMEOUT * 1000)
        page.set_viewport_size({"width": 1080, "height": 1920})
        logging.info("Navigating to %r", access_url)

        # Record HTTP status and navigated URLs so we can check errors later
        url_status = dict()
        navigated_urls = []
        page.on("response", lambda r: url_status.update({r.url: r.status}))
        page.on(
            "framenavigated",
            lambda f: f.parent_frame is None and navigated_urls.append(f.url),
        )

        # In fast mode use a shorter initial wait strategy (domcontentloaded) and only
        # escalate to full networkidle if needed.
        try:
            page.goto(access_url)
        except PlaywrightTimeoutError:
            logging.warning("Initial navigation timeout")

        try:
            if args.fast:
                page.wait_for_load_state("domcontentloaded")
            else:
                page.wait_for_load_state("networkidle")
        except PlaywrightTimeoutError:
            logging.warning(
                "Load state wait timed out (%s mode)", "fast" if args.fast else "normal"
            )

        # Check HTTP errors
        for url in navigated_urls:
            if (status_code := url_status.get(url, 0)) >= 400:
                error_cleanup(f"Got HTTP error {status_code}")

        # Early heuristic: if fast mode and initial text doesn't look like a policy,
        # skip readability injection to save 3 network fetches.
        page.evaluate("window.stop()")
        initial_text = page.content()
        looks_like_policy = (
            re.search(r"(data|privacy)\s*(policy|notice)", initial_text, re.I)
            is not None
        )
        if not args.fast or looks_like_policy:
            if not args.no_readability_js:
                page.add_script_tag(content=get_readability_js())
        readability_info = page.evaluate(
            r"""(no_readability_js) => {
            window.stop();

            const documentClone = document.cloneNode(true);
            const article = new Readability(documentClone).parse();
            if (!article) {
                throw new Error("Readability.js failed to parse the document");
            }
            article.applied = false;

            document.querySelectorAll('[aria-hidden=true]').forEach((x) => x.setAttribute("aria-hidden", false));

            if (isProbablyReaderable(document) && !no_readability_js) {
                documentClone.body.innerHTML = article.content;

                if (documentClone.body.innerText.search(/(data|privacy|cookie)\s*(policy|notice)/) >= 0) {
                    document.body.innerHTML = article.content;
                    article.applied = true;
                }
            }

            for (const elem of document.querySelectorAll('script, link, style, header, footer, nav'))
                elem.remove();

            return article;
        }""",
            [args.no_readability_js],
        )
        cleaned_html = page.content()

        # Check language
        soup = bs4.BeautifulSoup(cleaned_html, "lxml")
        soup_text = soup.body.text if soup.body else ""

        try:
            lang = langdetect.detect(soup_text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "UNKNOWN"

        if not lang.lower().startswith("en"):
            error_cleanup(f"Content language {lang} isn't English")

        if re.search(r"(data|privacy)\s*(?:policy|notice)", soup_text, re.I) is None:
            error_cleanup("Not like a privacy policy")

        # obtain the accessibility tree
        snapshot = page.accessibility.snapshot(interesting_only=False)

        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        with open(
            output_dir / "accessibility_tree.json", "w", encoding="utf-8"
        ) as fout:
            json.dump(snapshot, fout)

        with open(output_dir / "cleaned.html", "w", encoding="utf-8") as fout:
            fout.write(cleaned_html)

        with open(output_dir / "readability.json", "w", encoding="utf-8") as fout:
            json.dump(readability_info, fout)

        logging.info(
            "Saved to %s (elapsed %.2fs, fast=%s)",
            output_dir,
            time.monotonic() - t0,
            args.fast,
        )
        context.close()
        browser.close()


if __name__ == "__main__":
    # CLI with optional --fast flag
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (reduced waits, heuristic readability skip)",
    )
    cli_args = parser.parse_args()
    main(cli_args.url, cli_args.output_dir, fast=cli_args.fast)
