from datetime import date
from enum import Enum
import os

from bs4 import BeautifulSoup
import pymupdf4llm

class DocumentCaptureSource(Enum):
    PDF = "pdf"
    WEBPAGE = "webpage"

class GraphKind(Enum):
    STANDARD = "standard"
    LLM = "llm"
    NONE = "none"

class PolicyDocumentInfo:
    path: str
    output_dir: str
    source: DocumentCaptureSource
    capture_date: date
    has_results: bool = False
    def __init__(self, path: str, output_dir: str, source: DocumentCaptureSource, capture_date: date, has_results: bool):
        self.path = path
        self.output_dir = output_dir
        self.source = source
        self.capture_date = capture_date
        self.has_results = has_results

    def has_graph(self) -> bool:
        yml_path = os.path.join(self.output_dir, "graph-original.yml")
        return os.path.exists(self.output_dir) and os.path.exists(yml_path)

    def has_image(self) -> bool:
        png_path = os.path.join(self.output_dir, "knowledge_graph.png")
        return os.path.exists(self.output_dir) and os.path.exists(png_path)

    def _extract_text_from_webpage(self, path: str) -> str:
        policy_text = ""
        for fname in os.listdir(path):
            if fname.endswith(".html"):
                fpath = os.path.join(path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        for tag in soup.find_all(
                            ["p", "li", "h1", "h2", "h3", "h4", "div"]
                        ):
                            policy_text += tag.get_text() + "\n"
                except Exception:
                    continue
        return policy_text

    def _extract_text_from_pdf(self, path: str) -> str:
        # get first pdf item in folder
        pdf_path = None
        for fname in os.listdir(path):
            if fname.endswith(".pdf"):
                pdf_path = os.path.join(path, fname)
                break

        if not pdf_path:
            raise FileNotFoundError("No PDF file found")

        policy_text = ""
        doc = pymupdf4llm.pymupdf.open(pdf_path)
        for page in doc:
            text = page.get_text()
            policy_text += text + "\n"
        return policy_text

    def get_document_text(self) -> str:
        """Extract text from the document."""
        if self.source == DocumentCaptureSource.PDF:
            return self._extract_text_from_pdf(self.output_dir)
        elif self.source == DocumentCaptureSource.WEBPAGE:
            return self._extract_text_from_webpage(self.output_dir)
        return ""

class PolicyAnalysisResult:
    document: PolicyDocumentInfo
    score: float
    kind: GraphKind
    def __init__(self, document: PolicyDocumentInfo, score: float, kind: GraphKind):
        self.document = document
        self.score = score
        self.kind = kind

    def get_graph_image_path(self) -> str:
        """Get the file path to the graph image."""
        return os.path.join(self.document.output_dir, "knowledge_graph.png")

class PolicyDocumentProvider:
    def __init__(self, name: str, industry: str = "Unknown", documents: list[PolicyDocumentInfo] = None, results: list[PolicyAnalysisResult] = None):
        self.name = name
        self.documents = documents if documents is not None else []
        self.results = results if results is not None else []
        self.industry = industry
        self.output_dir = f"./output/{name.replace(' ', '_')}"

    def add_document(self, document: PolicyDocumentInfo):
        self.documents.append(document)

    def add_result(self, result: PolicyAnalysisResult):
        self.results.append(result)
