"""Serve the GlucoTwin home page (home/home.html) as the dashboard's first section.

home.html is a self-contained landing page in the design system (flowing hero, scroll-lit statements, three data-layer
tiles, the Bed 7 simulator, research log). Opened directly in a browser it works on its own. Inside Streamlit it runs
as a minimal custom component so it can (1) report its height, letting the iframe grow instead of scrolling, and
(2) send a value back when the visitor presses "Open the dashboard". The local Aspekta / Roboto Mono fonts from
static/ are inlined as data URIs because the component iframe cannot resolve relative font paths.
"""
from __future__ import annotations

import base64
from pathlib import Path

from . import config as C

HOME_PAGE = C.ROOT / "home" / "home.html"
STATIC_DIR = C.ROOT / "static"
COMPONENT_DIR = C.DATA_DIR / "artifacts" / "home"
_FONTS = (("Aspekta", "Aspekta-400.woff2"), ("Roboto Mono", "RobotoMono-400.woff2"))

# Custom-component handshake (component API v1): announce readiness, keep the iframe exactly as tall as the content,
# and expose window.gtGoto(page) so links inside the page can switch the dashboard section.
# The height is measured to the bottom of the page footer, not documentElement.scrollHeight: scrollHeight can never be
# smaller than the iframe itself, so once the iframe grew it could not shrink back and left a blank band after the footer.
_EMBED_HEAD = '<script>window.GT_EMBED = true; document.documentElement.classList.add("embed");</script>'
_HANDSHAKE = ('<script>(() => { const post = (type, extra) => parent.postMessage({isStreamlitMessage: true, type, '
              '...extra}, "*"); post("streamlit:componentReady", {apiVersion: 1}); const send = () => { '
              'const f = document.querySelector("footer"); const height = f ? Math.ceil(f.getBoundingClientRect().bottom '
              '+ window.scrollY) : document.documentElement.scrollHeight; post("streamlit:setFrameHeight", {height}); }; '
              'new ResizeObserver(send).observe(document.body); addEventListener("load", send); '
              'if (document.fonts) document.fonts.ready.then(send); '
              'window.gtGoto = page => post("streamlit:setComponentValue", {value: {page, n: Date.now()}, dataType: "json"}); '
              '})();</script>')


def _swap(html: str, old: str, new: str) -> str:
    if html.count(old) != 1:
        raise ValueError(f"home.html no longer contains {old!r} exactly once; update src/home.py")
    return html.replace(old, new)


def home_available() -> bool:
    return HOME_PAGE.exists()


def _font_css() -> str:
    rules = []
    for family, name in _FONTS:
        path = STATIC_DIR / name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            rules.append(f'@font-face{{font-family:"{family}";src:url("data:font/woff2;base64,{data}") format("woff2");'
                         f'font-weight:400;font-style:normal;font-display:swap}}')
    return "<style>" + "".join(rules) + "</style>"


def home_html() -> str:
    page = HOME_PAGE.read_text(encoding="utf-8")
    page = _swap(page, "<head>\n", "<head>\n" + _EMBED_HEAD + "\n" + _font_css() + "\n")
    return _swap(page, "</body>", _HANDSHAKE + "\n</body>")


def home_component_dir() -> Path:
    """Write the embeddable page to COMPONENT_DIR/index.html (only when it changed) and return the folder."""
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    index, html = COMPONENT_DIR / "index.html", home_html()
    if not index.exists() or index.read_text(encoding="utf-8") != html:
        index.write_text(html, encoding="utf-8")
    return COMPONENT_DIR
