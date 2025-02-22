#!/usr/bin/env python3
"""Convert an imported PDF to a HTML document for parsing."""

import argparse
from pdf2docx import Converter
from bs4 import BeautifulSoup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Input path")
    parser.add_argument("output", help="Output dir")
    args = parser.parse_args()

    # Convert PDF to DOCX
    cv = Converter(args.path)
    cv.convert(args.output / "policy.docx", start=0, end=None)
    cv.close()

    # Convert DOCX to HTML
    with open(args.output / "policy.docx", "rb") as docx_file:
        soup = BeautifulSoup(docx_file, "html.parser")

    with open(args.output / "policy.html", "w") as html_file:
        html_file.write(str(soup))


if __name__ == "__main__":
    main()
