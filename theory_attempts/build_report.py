"""Offline MathML HTML/PDF export when local MiKTeX cannot initialize.

Needs Pandoc; PDF additionally needs Edge or Chrome. The browser runs headless
with its temporary profile inside theory_attempts/.build. No package download
or changes to the user's TeX installation are performed.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--stem", default="bounds",
                        help="Report filename without extension, inside theory_attempts")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.stem):
        parser.error("--stem must be a plain filename stem")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Pandoc is needed for the HTML fallback; bounds.tex remains available")
    build = HERE / ".build"
    build.mkdir(exist_ok=True)
    original = (HERE / (args.stem + ".tex")).read_text(encoding="utf-8")
    source = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{[^}]+)\.pdf(\})",
                    lambda m: m.group(1) + ".png" + m.group(2), original)
    # Pandoc does not preserve hyperref's \path command; export it as code.
    source = re.sub(r"\\path\{([^}]+)\}", lambda m: r"\verb|" + m.group(1) + "|", source)
    web_tex = build / (args.stem + "_web.tex")
    web_tex.write_text(source, encoding="utf-8")
    html_path = HERE / (args.stem + ".html")
    subprocess.run([pandoc, str(web_tex), "--from=latex", "--to=html5", "--standalone",
                    "--mathml", "--embed-resources", "--resource-path=.",
                    "--css=report.css", "--output=" + str(html_path)],
                   cwd=str(HERE), check=True)
    html = html_path.read_text(encoding="utf-8")

    # Pandoc preserves theorem references but does not number LaTeX equation
    # references. Match display blocks in source order, retain every equation,
    # and supply explicit anchors/numbers without changing mathematical text.
    source_blocks = list(re.finditer(
        r"\\begin\{equation\}.*?\\end\{equation\}|"
        r"\\begin\{align\}.*?\\end\{align\}|\\\[.*?\\\]", source, re.S))
    html_blocks = list(re.finditer(r'<math display="block".*?</math>', html, re.S))
    if len(source_blocks) != len(html_blocks):
        raise RuntimeError("Display block counts differ; inspect the HTML export before using it")
    labels, replacements = {}, []
    number = 0
    for src, dest in zip(source_blocks, html_blocks):
        names = re.findall(r"\\label\{([^}]+)\}", src.group(0))
        if not names:
            continue
        nums = []
        for name in names:
            number += 1
            labels[name] = number
            nums.append("(" + str(number) + ")")
        anchors = "".join('<span id="' + name + '"></span>' for name in names)
        replacement = ('<div class="equation">' + anchors + dest.group(0)
                       + '<span class="eqno">' + ", ".join(nums) + '</span></div>')
        replacements.append((dest.start(), dest.end(), replacement))
    for start, end, replacement in reversed(replacements):
        html = html[:start] + replacement + html[end:]
    for label, num in labels.items():
        pattern = (r'(<a href="#' + re.escape(label)
                   + r'"[^>]*data-reference-type="eqref"[^>]*>).*?(</a>)')
        html = re.sub(pattern, lambda m: m.group(1) + "(" + str(num) + ")" + m.group(2), html)
    unresolved = re.findall(r'data-reference-type="eqref"[^>]*>\[([^<]+)\]', html)
    if unresolved:
        raise RuntimeError("Unresolved equation references: " + repr(unresolved))
    html_path.write_text(html, encoding="utf-8")
    print("Wrote", html_path.name, "with", len(html_blocks), "MathML displays")
    if args.html_only:
        return
    candidates = [
        shutil.which("msedge"), shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser = next((p for p in candidates if p and Path(p).is_file()), None)
    if not browser:
        print("No browser found for PDF export; the standalone HTML is ready")
        return
    pdf = HERE / (args.stem + ".pdf")
    result = subprocess.run([
        browser, "--headless", "--disable-gpu", "--disable-software-rasterizer",
        "--no-sandbox", "--disable-gpu-sandbox", "--no-first-run", "--no-default-browser-check",
        "--no-pdf-header-footer", "--allow-file-access-from-files",
        "--user-data-dir=" + str(build / "browser-profile"),
        "--print-to-pdf=" + str(pdf), html_path.as_uri(),
    ], cwd=str(HERE), capture_output=True, text=True, timeout=60)
    (build / "browser-export.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("Browser PDF export failed; see .build/browser-export.log")
    print("Wrote", pdf.name, "(%d bytes)" % pdf.stat().st_size)


if __name__ == "__main__":
    main()
