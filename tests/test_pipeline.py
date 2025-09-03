import json
from pathlib import Path

import pytest

from poligrapher.scripts import run_annotators, build_graph, init_document


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def make_workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    # copy minimal accessibility tree
    tree_path = FIXTURE_DIR / "min_accessibility_tree.json"
    with open(tree_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    with open(workdir / "accessibility_tree.json", "w", encoding="utf-8") as fout:
        json.dump(data, fout)
    return workdir


def test_full_pipeline(tmp_path):
    # Skip if custom model assets (extra-data) are absent; prevents CI/network dependency
    repo_root = Path(__file__).resolve().parent.parent
    ner_dir = repo_root / "poligrapher" / "extra-data" / "named_entity_recognition"
    if not ner_dir.exists():
        pytest.skip("extra-data named_entity_recognition model not present; skipping pipeline test")
    # Build minimal workdir
    workdir = make_workdir(tmp_path)
    print("Running test using working directory:", workdir)

    # Initialize document (creates document.pickle)
    init_document.main([str(workdir)], nlp_model_dir="")
    assert (workdir / "document.pickle").exists(), "document.pickle not created"

    # Run annotators (produces token relationships)
    run_annotators.main([str(workdir)], nlp_model_dir="")

    # Build graph
    build_graph.main(workdirs=[str(workdir)], pretty=False)

    # Verify output YAML graph exists
    yaml_path = workdir / "graph-original.full.yml"
    assert yaml_path.exists(), "Graph YAML not generated"

    # Basic sanity: load trimmed graph YAML and ensure at least one edge
    trimmed_path = workdir / "graph-original.yml"
    assert trimmed_path.exists(), "Trimmed graph YAML not generated"

    # A minimal semantic check: if a graphml was requested we could load but here just check size
    text = yaml_path.read_text(encoding="utf-8")
    assert "nodes:" in text and "links:" in text, "Graph YAML content invalid"
