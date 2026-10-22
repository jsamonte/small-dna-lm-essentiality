"""Render paper/manuscript.md to a submittable PDF.

bioRxiv wants a PDF; the build produces Markdown. This converts one to the
other without a LaTeX toolchain: Markdown to HTML, a print stylesheet, then
headless Chrome's own print-to-PDF. Chrome is already on this machine and its
renderer is the same one that lays out the HTML preview, so what you proof is
what you submit.

Deliberately not pandoc or LaTeX: both are large installs, and neither earns
its keep for a manuscript that is prose, tables and one figure.

    .venv\\Scripts\\python scripts\\build_pdf.py

The script refuses to run if any section still carries the <!-- DRAFT --> or
<!-- UNWRITTEN --> marker, unless --force is given. A PDF is the thing you
hand to other people, and handing over a generated outline or prose that is
not yours is the specific accident worth making difficult.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SOURCE = PAPER / "manuscript.md"
OUT = PAPER / "manuscript.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

STYLE = """
@page { size: Letter; margin: 22mm 20mm; }
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.55; color: #111; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 17pt; line-height: 1.22; margin: 0 0 3mm; letter-spacing: -0.01em; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm; padding-bottom: 1mm;
     border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: justify; hyphens: auto; }
em { color: #333; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
blockquote { margin: 3mm 0; padding: 2mm 0 2mm 4mm; border-left: 2.5px solid #bbb;
             color: #444; font-style: italic; break-inside: avoid; }
code { font-family: "IBM Plex Mono", Consolas, monospace; font-size: 9pt;
       background: #f3f2ee; padding: 0.3mm 1mm; }
pre { font-family: "IBM Plex Mono", Consolas, monospace; font-size: 8.6pt;
      background: #f3f2ee; padding: 2.5mm 3mm; line-height: 1.42;
      white-space: pre-wrap; break-inside: avoid; border-left: 2px solid #ccc; }
pre code { background: none; padding: 0; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm;
        font-size: 9.1pt; break-inside: avoid; }
th { text-align: left; font-size: 7.8pt; text-transform: uppercase;
     letter-spacing: 0.06em; color: #666; font-weight: 600;
     border-bottom: 1.2px solid #aaa; padding: 1.6mm 2mm; vertical-align: bottom; }
td { padding: 1.7mm 2mm; border-bottom: 1px solid #ddd; vertical-align: top; }
tr:last-child td { border-bottom: none; }
figure { margin: 5mm 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 8.8pt; color: #444; margin-top: 2mm;
             text-align: left; line-height: 1.45; }
hr { border: none; border-top: 1px solid #ccc; margin: 5mm 0; }
"""


def find_chrome():
    for path in CHROME_CANDIDATES:
        if path and Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit(
        "Could not find Chrome or Edge. Install one, or edit CHROME_CANDIDATES.")


def unfinished(text):
    """Section markers that must not reach a PDF."""
    found = []
    if "<!-- DRAFT -->" in text:
        found.append("<!-- DRAFT --> (prose is Claude's, not yours)")
    if "<!-- UNWRITTEN -->" in text:
        found.append("<!-- UNWRITTEN --> (section is still a template)")
    if re.search(r">>>.*<<<", text):
        found.append(">>> ... <<< (an instruction left in the text)")
    return found


def to_html(md_text, title):
    try:
        import markdown
    except ImportError:
        raise SystemExit("pip install markdown")
    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    # Markdown leaves bare <img> paragraphs; promote them to figures so the
    # caption that follows binds to the image and they page-break together.
    body = re.sub(r"<p>(<img[^>]*>)</p>", r"<figure>\1</figure>", body)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>{STYLE}</style></head>"
            f"<body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="render even if sections are unfinished")
    ap.add_argument("--open", action="store_true", help="open the PDF when done")
    args = ap.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found - run build_manuscript.py first")
    text = SOURCE.read_text(encoding="utf-8")

    problems = unfinished(text)
    if problems and not args.force:
        print("REFUSING to build a PDF. The manuscript still contains:")
        for p in problems:
            print(f"  - {p}")
        print("\nA PDF is what you hand to other people. Finish the sections,")
        print("or pass --force if you need a proof copy for yourself.")
        return 1
    if problems:
        print("  (--force) building anyway, with:")
        for p in problems:
            print(f"    ! {p}")

    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_match.group(1).strip() if title_match else ROOT.name

    html = to_html(text, title)
    with tempfile.TemporaryDirectory() as tmp:
        # Written beside the manuscript so relative figure paths resolve.
        staged = PAPER / "_manuscript_render.html"
        staged.write_text(html, encoding="utf-8")
        try:
            subprocess.run(
                [find_chrome(), "--headless", "--disable-gpu",
                 "--no-pdf-header-footer", f"--user-data-dir={tmp}",
                 f"--print-to-pdf={OUT}", staged.as_uri()],
                check=False, capture_output=True, timeout=180)
        finally:
            staged.unlink(missing_ok=True)

    if not OUT.exists() or OUT.stat().st_size < 5000:
        raise SystemExit("Chrome produced no usable PDF.")
    print(f"  wrote {OUT.relative_to(ROOT)}  "
          f"({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  title: {title}")
    if args.open:
        os.startfile(OUT)  # noqa: S606
    return 0


if __name__ == "__main__":
    sys.exit(main())
