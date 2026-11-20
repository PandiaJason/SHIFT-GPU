"""
Compiles paper/FULL_PAPER.md into an academic publication-grade PDF using Pandoc and Headless Chrome.
Produces paper/SHIFT_GPU_Full_Paper.pdf and copies to workspace root.
"""

import os
import subprocess
import shutil

MD_PATH = "paper/FULL_PAPER.md"
HTML_PATH = "paper/paper_styled.html"
PDF_PATH = "paper/SHIFT_GPU_Full_Paper.pdf"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 22mm 20mm 25mm 20mm;
    @top-center {
        content: "Sustainable Computing: Informatics and Systems (Elsevier)";
        font-family: 'Crimson Pro', serif;
        font-size: 8pt;
        color: #777;
        font-style: italic;
    }
    @bottom-center {
        content: counter(page);
        font-family: 'Crimson Pro', serif;
        font-size: 9pt;
        color: #555;
    }
}

body {
    font-family: 'Crimson Pro', Georgia, 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.48;
    color: #1a1a1a;
    text-align: justify;
    margin: 0;
    padding: 0;
}

h1.title {
    font-size: 20pt;
    font-weight: 700;
    text-align: center;
    line-height: 1.25;
    margin-bottom: 8px;
    color: #0d233a;
}

.authors-block {
    text-align: center;
    font-size: 11pt;
    margin-bottom: 20px;
    color: #444;
}

.journal-header {
    text-align: center;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8c1d40;
    font-weight: 600;
    margin-bottom: 12px;
}

h1 {
    font-size: 14pt;
    font-weight: 700;
    border-bottom: 1px solid #0d233a;
    padding-bottom: 3px;
    margin-top: 22px;
    margin-bottom: 10px;
    color: #0d233a;
    page-break-after: avoid;
}

h2 {
    font-size: 12pt;
    font-weight: 700;
    margin-top: 16px;
    margin-bottom: 6px;
    color: #1a365d;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    margin-top: 12px;
    margin-bottom: 4px;
    color: #2b6cb0;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
    text-indent: 1.5em;
}

p:first-of-type, h1 + p, h2 + p, h3 + p, .no-indent {
    text-indent: 0;
}

.abstract-box {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0d233a;
    padding: 14px 18px;
    margin: 18px 0 24px 0;
    font-size: 9.5pt;
    line-height: 1.42;
}

.abstract-box strong {
    color: #0d233a;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    padding: 6px 10px;
    text-align: left;
}

th {
    background: #edf2f7;
    font-weight: 700;
    border-top: 1.5px solid #2d3748;
    border-bottom: 1.5px solid #2d3748;
    color: #1a202c;
}

tr:nth-child(even) td {
    background: #fcfdfe;
}

td {
    border-bottom: 1px solid #e2e8f0;
}

tr:last-child td {
    border-bottom: 1.5px solid #2d3748;
}

code, pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5pt;
}

pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
    page-break-inside: avoid;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto;
}

.caption {
    font-size: 8.5pt;
    color: #555;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 14px;
    font-style: italic;
}

hr {
    border: 0;
    border-top: 1px solid #cbd5e0;
    margin: 20px 0;
}

ul, ol {
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 24px;
}

li {
    margin-bottom: 3px;
}
</style>
"""

def generate_pdf():
    print("Reading paper markdown...")
    with open(MD_PATH, "r") as f:
        md_content = f.read()

    # Pandoc HTML compilation
    print("Converting Markdown to standalone HTML via Pandoc...")
    cmd_pandoc = [
        "pandoc",
        MD_PATH,
        "-f", "markdown+tex_math_dollars+raw_html",
        "-t", "html5",
        "--standalone",
        "--mathjax",
        "-o", HTML_PATH
    ]
    subprocess.run(cmd_pandoc, check=True)

    # Inject academic styles
    with open(HTML_PATH, "r") as f:
        html = f.read()

    html = html.replace("</head>", f"{CSS_STYLES}\n</head>")

    with open(HTML_PATH, "w") as f:
        f.write(html)

    print("Rendering HTML to PDF via Headless Google Chrome...")
    abs_html = os.path.abspath(HTML_PATH)
    abs_pdf = os.path.abspath(PDF_PATH)
    cmd_chrome = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={abs_pdf}",
        f"file://{abs_html}"
    ]
    subprocess.run(cmd_chrome, check=True)

    # Copy to root directory for easy access
    shutil.copy(PDF_PATH, "SHIFT_GPU_Full_Paper.pdf")
    print(f"\nPDF successfully generated: {PDF_PATH}")
    print("A copy has been placed at workspace root: SHIFT_GPU_Full_Paper.pdf")

if __name__ == "__main__":
    generate_pdf()
