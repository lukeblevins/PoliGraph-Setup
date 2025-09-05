"""Policy analysis domain entities used by the Gradio app.

Historically the UI + experimental notebooks accessed policy metadata, file
paths, and derived scores by passing around raw rows from one or more CSV
files (e.g., columns like `provider`, `policy_path`, `score`, flags, etc.).
That approach caused several problems:

1. Tight coupling / duplication: Every place that needed policy data re‑built
    file system paths (often slightly differently) and re‑implemented checks
    like "does the graph image exist?".
2. Brittle refactors: Renaming a column or changing artifact names required
    hunting through ad‑hoc DataFrame code across the app.
3. Implicit semantics: There was no obvious, typed distinction between a
    provider (company), a *captured* document (PDF vs scraped HTML bundle), and
    an *analysis result* (score associated with a chosen graph representation).
4. Mixed concerns: Text extraction logic sometimes leaked into UI callbacks.

This module replaces the ad‑hoc CSV row passing pattern with light‑weight
entity classes that encapsulate behavior, hide path conventions, and make the
intent explicit. Enumerations (`DocumentCaptureSource`, `GraphKind`) provide
closed sets instead of free‑form strings, reducing silent typos.

Key concepts:
* PolicyDocumentInfo  -> a concrete captured policy (folder of artifacts) and
                                 utilities to extract its raw text.
* PolicyAnalysisResult -> an analysis *outcome* (e.g., score produced from a
                                  particular graph flavor) tied back to a document.
* PolicyDocumentProvider -> groups multiple documents + results for a single
                                     organization (company) and centralizes its output
                                     directory naming.

When adding new artifact types or analysis dimensions, extend the relevant
entity (e.g., another helper on PolicyDocumentInfo) rather than sprinkling new
CSV columns + path logic across the UI. This keeps downstream code declarative
and easier to test.
"""

from datetime import date
from enum import Enum
import os

from bs4 import BeautifulSoup
import pymupdf4llm

class DocumentCaptureSource(Enum):
    """Origin / capture modality of a policy.

    Replaces prior string literals (e.g., 'pdf', 'html', 'web') scattered in
    CSV rows. Using an Enum gives IDE autocomplete + guards against typos.
    """

    PDF = "pdf"
    WEBPAGE = "webpage"

class GraphKind(Enum):
    """Type of knowledge graph / extraction variant used for scoring.

    STANDARD: the traditional pipeline graph (token + annotator edges).
    LLM: an alternative / experimental graph derived via LLM summarization.
    NONE: placeholder when a score exists independent of a graph variant.
    """

    STANDARD = "standard"
    LLM = "llm"
    NONE = "none"

class PolicyDocumentInfo:
    """Encapsulates a single captured policy artifact set.

    Attributes
    -----------
    path: Original source path (file system path for PDF or root of scraped
          HTML bundle). Maintained for provenance.
    output_dir: Working directory produced by the pipeline (contains
                accessibility_tree.json, cleaned.html, document.pickle,
                graph-original.yml, knowledge_graph.png, etc.). Centralized so
                callers don't rebuild paths manually.
    source: Capture modality (Enum) -> replaces earlier free‑form CSV field.
    capture_date: Date the policy was captured (previously often stored as a
                  string column). Use a date object to make comparisons &
                  formatting explicit.
    has_results: Convenience flag signaling if any analyses have been run.

    Behavior
    --------
    Provides helper predicates (has_graph / has_image) and text extraction
    logic that used to live inside UI callbacks or notebook cells.
    """

    path: str
    output_dir: str
    source: DocumentCaptureSource
    capture_date: date
    has_results: bool = False

    def __init__(self, path: str, output_dir: str, source: DocumentCaptureSource, capture_date: date, has_results: bool):
        self.path = path  # Raw source path for traceability
        self.output_dir = output_dir  # Pipeline workdir root
        self.source = source
        self.capture_date = capture_date
        self.has_results = has_results

    def has_graph(self) -> bool:
        """True if a graph YAML exists for this document.

        Centralizes canonical filename ("graph-original.yml") so changes only
        occur here vs. many string literals previously embedded in CSV logic.
        """
        yml_path = os.path.join(self.output_dir, "graph-original.yml")
        return os.path.exists(self.output_dir) and os.path.exists(yml_path)

    def has_image(self) -> bool:
        """True if the rendered graph image artifact exists."""
        png_path = os.path.join(self.output_dir, "knowledge_graph.png")
        return os.path.exists(self.output_dir) and os.path.exists(png_path)

    def _extract_text_from_webpage(self, path: str) -> str:
        """Aggregate visible text from stored HTML files.

        Previously: code in notebooks iterated over a directory, parsed each
        HTML with BeautifulSoup, and concatenated <p> text inline. Encapsulate
        that once so consumers just call get_document_text().
        """
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
        """Extract raw text from the first PDF in the folder.

        Consolidates PDF parsing logic (previously replicated where needed).
        If multiple PDFs ever need support, evolve the selection rule here
        without touching callers.
        """
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
        """Unified text accessor (source‑aware).

        Hides modality branching so calling code does not duplicate if/else
        logic. Returns empty string on unsupported sources for resilience.
        """
        if self.source == DocumentCaptureSource.PDF:
            return self._extract_text_from_pdf(self.output_dir)
        elif self.source == DocumentCaptureSource.WEBPAGE:
            return self._extract_text_from_webpage(self.output_dir)
        return ""

class PolicyAnalysisResult:
    """Represents the outcome of analyzing a document with a specific graph kind.

    Encapsulates (document ↔ score ↔ graph variant) triple that used to be
    spread across multiple CSV columns. Keeping this as a distinct object lets
    us associate future metadata (confidence intervals, method version, etc.)
    without schema churn across the UI.
    """

    document: PolicyDocumentInfo
    score: float
    kind: GraphKind

    def __init__(self, document: PolicyDocumentInfo, score: float, kind: GraphKind):
        self.document = document
        self.score = score
        self.kind = kind

    def get_graph_image_path(self) -> str:
        """Return the canonical graph image path for the linked document."""
        return os.path.join(self.document.output_dir, "knowledge_graph.png")

class PolicyDocumentProvider:
    """Represents a single organization (company) that owns policies.

    Collates its documents and analysis results. Previously, grouping logic
    lived in Pandas groupby chains over provider columns. Centralizing here
    gives a single place to evolve naming (e.g., output directory slug rules)
    and attach provider metadata (industry, region, risk tier, etc.).
    """

    def __init__(self, name: str, industry: str = "Unknown", documents: list[PolicyDocumentInfo] = None, results: list[PolicyAnalysisResult] = None):
        self.name = name
        self.documents = documents if documents is not None else []
        self.results = results if results is not None else []
        self.industry = industry
        # Central slug formatting; previously duplicated with subtle variations.
        self.output_dir = f"./output/{name.replace(' ', '_')}"

    def add_document(self, document: PolicyDocumentInfo):
        """Attach a new captured policy to this provider."""
        self.documents.append(document)

    def add_result(self, result: PolicyAnalysisResult):
        """Record an analysis result for one of the provider's documents."""
        self.results.append(result)
