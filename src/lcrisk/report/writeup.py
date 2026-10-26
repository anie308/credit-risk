"""Build the model documentation: Markdown (docs/model_doc) -> HTML -> PDF via headless Chrome.

There is no LaTeX toolchain here, so the document is styled HTML printed by Chrome or Edge.
Figures come from `lcrisk figures` and are copied next to the HTML.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path

import markdown

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]

CSS = """
@page { size: A4; margin: 20mm 19mm 20mm 19mm; }
html { font-family: "Georgia", "Times New Roman", serif; font-size: 10.2pt; line-height: 1.42;
       color: #1a1a1a; }
body { max-width: 172mm; margin: 0 auto; }
header { text-align: center; margin-bottom: 1.2em; }
header h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 0.4em; font-weight: normal; }
header .meta { font-size: 10pt; color: #444; }
header .status { font-size: 8.5pt; color: #8a3b22; margin-top: 0.3em; }
h2 { font-size: 12.5pt; margin: 1.3em 0 0.4em; break-after: avoid; }
h3 { font-size: 11pt; font-style: italic; font-weight: normal; margin: 1em 0 0.3em;
     break-after: avoid; }
p { margin: 0 0 0.6em; text-align: justify; hyphens: auto; }
h2#abstract + p { font-size: 9.8pt; }
blockquote { margin: 0.5em 2em; font-style: italic; }
figure { margin: 0.6em 0 0.8em; break-inside: avoid; text-align: center; }
figure img { max-width: 74%; }
p:has(> img) { text-align: center; margin: 0.6em 0 0.2em; break-inside: avoid;
               break-after: avoid; }
p > img { max-width: 80%; max-height: 92mm; }
p:has(> img) + p > em:only-child { font-size: 8.8pt; color: #333; }
figcaption { font-size: 8.8pt; color: #333; text-align: left; margin-top: 0.3em; }
table { border-collapse: collapse; margin: 0.6em auto 1em; font-size: 9.5pt;
        break-inside: avoid; }
th, td { padding: 0.2em 0.8em; border-bottom: 1px solid #ccc; text-align: left; }
th { border-bottom: 1.5px solid #333; font-weight: bold; }
ul { margin: 0 0 0.6em; padding-left: 1.3em; }
li { margin-bottom: 0.25em; }
code { font-family: Consolas, monospace; font-size: 9pt; }
.note { font-size: 8.8pt; color: #555; border-top: 1px solid #ccc; padding-top: 0.5em;
        margin-top: 1.5em; }
"""


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    """'---' key: value lines '---' at the top -> (fields, body)."""
    if not text.startswith("---"):
        return {}, text
    head, body = text[3:].split("\n---", 1)
    fields = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, body.lstrip("\n")


def render_html(source: Path) -> str:
    meta, body = _front_matter(source.read_text(encoding="utf-8"))
    content = markdown.markdown(body, extensions=["tables", "toc"])
    status = f'<div class="status">{html.escape(meta["status"])}</div>' if "status" in meta else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(meta.get("title", ""))}</title>
<style>{CSS}</style></head>
<body>
<header><h1>{html.escape(meta.get("title", ""))}</h1>
<div class="meta">{html.escape(meta.get("author", ""))} · {html.escape(meta.get("date", ""))}</div>
{status}</header>
{content}
</body></html>
"""


def find_browser() -> str:
    if env := os.environ.get("LCRISK_BROWSER"):
        return env
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise FileNotFoundError("no Chrome/Edge found; set LCRISK_BROWSER to its path")


def build(source: Path, figures: Path, out: Path) -> tuple[Path, Path]:
    """Write <out>/<stem>.html (+ figures/) and print it to <out>/<stem>.pdf."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(figures, out / "figures", dirs_exist_ok=True)
    page = out / f"{source.stem}.html"
    page.write_text(render_html(source), encoding="utf-8")
    pdf = out / f"{source.stem}.pdf"
    subprocess.run([
        find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--print-to-pdf-no-header", f"--print-to-pdf={pdf.resolve()}",
        page.resolve().as_uri(),
    ], check=True, capture_output=True, timeout=120)
    if not pdf.exists():
        raise RuntimeError(f"browser exited without writing {pdf}")
    return page, pdf
