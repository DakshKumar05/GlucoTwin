"""Embed the standalone "Bed 7 Patient Journey" teaching page in the dashboard.

The page lives outside this project, next to it: ../demo/patient-journey.html with ../styles/tokens.css and
../fonts/Aspekta-400.woff2. A Streamlit iframe cannot resolve those relative paths, so `journey_html` inlines the
stylesheet and the font (as a data URI). It is served as a minimal custom component (`journey_component_dir`) because
only custom components may report their height, which lets the iframe grow to fit instead of scrolling inside the
dashboard. The source page itself is never modified.
"""
from __future__ import annotations

import base64
from pathlib import Path

from . import config as C

JOURNEY_ROOT = C.ROOT.parent
JOURNEY_PAGE = JOURNEY_ROOT / "demo" / "patient-journey.html"
TOKENS_CSS = JOURNEY_ROOT / "styles" / "tokens.css"
ASPEKTA_FONT = JOURNEY_ROOT / "fonts" / "Aspekta-400.woff2"
COMPONENT_DIR = C.DATA_DIR / "artifacts" / "patient_journey"

# Custom-component handshake (component API v1): announce readiness, then keep the iframe as tall as the page.
_FIT_HEIGHT = ('<script>(() => { const post = (type, extra) => parent.postMessage({isStreamlitMessage: true, type, '
               '...extra}, "*"); post("streamlit:componentReady", {apiVersion: 1}); const send = () => '
               'post("streamlit:setFrameHeight", {height: document.documentElement.scrollHeight}); '
               'new ResizeObserver(send).observe(document.documentElement); addEventListener("load", send); })();'
               '</script>')


def _swap(html: str, old: str, new: str) -> str:
    if html.count(old) != 1:
        raise ValueError(f"patient-journey.html no longer contains {old!r} exactly once; update src/journey.py")
    return html.replace(old, new)


def journey_available() -> bool:
    return all(p.exists() for p in (JOURNEY_PAGE, TOKENS_CSS, ASPEKTA_FONT))


def journey_html() -> str:
    """Return the patient-journey page as one self-contained HTML string, ready for an iframe."""
    page = JOURNEY_PAGE.read_text(encoding="utf-8")
    font = base64.b64encode(ASPEKTA_FONT.read_bytes()).decode("ascii")
    css = _swap(TOKENS_CSS.read_text(encoding="utf-8"), 'url("../fonts/Aspekta-400.woff2")',
                f'url("data:font/woff2;base64,{font}")')
    page = _swap(page, '<link rel="preload" href="../fonts/Aspekta-400.woff2" as="font" type="font/woff2" crossorigin>\n', "")
    page = _swap(page, '<link rel="stylesheet" href="../styles/tokens.css">', f"<style>\n{css}</style>")
    return _swap(page, "</body>", _FIT_HEIGHT + "\n</body>")


def journey_component_dir() -> Path:
    """Write the self-contained page to COMPONENT_DIR/index.html (only when it changed) and return the folder."""
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    index, html = COMPONENT_DIR / "index.html", journey_html()
    if not index.exists() or index.read_text(encoding="utf-8") != html:
        index.write_text(html, encoding="utf-8")
    return COMPONENT_DIR
