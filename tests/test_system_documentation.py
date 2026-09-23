from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "static" / "system-and-coverage.html"


class IframeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.iframes: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "iframe":
            self.iframes.append(dict(attrs))


def test_research_guide_documents_the_complete_experiment() -> None:
    html = GUIDE.read_text(encoding="utf-8")
    for heading in (
        "End-to-end data pipeline",
        "Run-type comparison",
        "Single Prompt",
        "Staged Single Agent",
        "Centralized Multi-Agent",
        "Evaluation methodology",
        "Result-affecting technology",
        "Interpretation and reproducibility limits",
    ):
        assert heading in html
    for policy in ("Gemini 3.6 Flash", "medium thinking", "100,000-token"):
        assert policy in html

    parser = IframeParser()
    parser.feed(html)
    expected = {
        "system-and-coverage-diagram.html",
        "single-prompt-flow.html",
        "staged-single-agent-flow.html",
        "centralized-multi-agent-flow.html",
    }
    assert {frame["src"] for frame in parser.iframes} == expected
    assert all(frame.get("title") for frame in parser.iframes)
    assert all((GUIDE.parent / asset).is_file() for asset in expected)
