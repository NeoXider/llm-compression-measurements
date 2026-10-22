"""Minimal Markdown -> LaTeX converter for the paper Markdown (headings, images with captions, paragraphs, lists, pipe tables,
indented code, **bold**, *italic*, `code`). Output is a starting point for arXiv; not compiled here."""
from __future__ import annotations

import re
import sys
from pathlib import Path

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNI = {"≈": r"$\approx$", "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "−": "--", "–": "--", "—": "---",
       "→": r"$\rightarrow$", "·": r"$\cdot$", "…": r"\ldots{}", "⁻³": r"$^{-3}$", "≠": r"$\neq$", "±": r"$\pm$",
       "§": r"\S{}", "“": "``","”": "''", "’": "'", "°": r"$^\circ$", "†": r"$\dagger$"}


def esc(s: str) -> str:
    s = "".join(SPECIAL.get(c, c) for c in s)
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def inline(s: str) -> str:
    parts = re.split(r"(`[^`]+`)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{" + esc(p[1:-1]) + "}")
        else:
            t = esc(p)
            t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", t)
            out.append(t)
    return "".join(out)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|?\s*-", r.strip().strip("|"))]
    n = max(len(r) for r in cells)
    spec = "l" * n
    lines = [r"\begin{table}[h]\centering\scriptsize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + spec + "}", r"\toprule"]
    for i, r in enumerate(cells):
        r = r + [""] * (n - len(r))
        lines.append(" & ".join(inline(c) for c in r) + r" \\")
        if i == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    lines = src.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    body, i = [], 1
    while i < len(lines):
        ln = lines[i]
        if i < 4 and ln.startswith("NeoXider — v3"):
            i += 1; continue
        if ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            body.append(table(lines[i:j])); i = j; continue
        mi = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", ln)
        if mi:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = ""
            if j < len(lines) and lines[j].startswith("**Figure"):
                cap = re.sub(r"^\*\*Figure [^*]+\*\*\s*", "", lines[j]); i = j + 1
            else:
                i += 1
            body.append(r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{" + mi.group(1) + "}"
                        + (r"\caption{" + inline(cap) + "}" if cap else "") + r"\end{figure}")
            continue
        if ln.startswith("    ") and ln.strip():
            j = i
            while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
                j += 1
            body.append(r"\begin{verbatim}" + "\n" + "\n".join(l[4:] for l in lines[i:j]).rstrip() + "\n" + r"\end{verbatim}"); i = j; continue
        m = re.match(r"^(#{2,4})\s+(.*)", ln)
        if m:
            lvl = {2: "section", 3: "subsection", 4: "subsubsection"}[len(m.group(1))]
            name = re.sub(r"^\d+(\.\d+)*\.?\s+", "", m.group(2))
            if name.lower().startswith("abstract"):
                body.append(r"\begin{abstract}"); i += 1
                while i < len(lines) and not lines[i].startswith("#"):
                    if lines[i].strip():
                        body.append(inline(lines[i]))
                    i += 1
                body.append(r"\end{abstract}"); continue
            if name.lower().startswith("appendix"):
                if r"\appendix" not in body:
                    body.append(r"\appendix")
                name = re.sub(r"^Appendix [A-Z]\.\s*", "", name)
            body.append(f"\\{lvl}{{{inline(name)}}}"); i += 1; continue
        if re.match(r"^\s*(-|\d+\.)\s+", ln):
            kind = "enumerate" if re.match(r"^\s*\d+\.", ln) else "itemize"
            items, j = [], i
            while j < len(lines) and (re.match(r"^\s*(-|\d+\.)\s+", lines[j]) or (lines[j].startswith("  ") and lines[j].strip())):
                if re.match(r"^\s*(-|\d+\.)\s+", lines[j]):
                    items.append(re.sub(r"^\s*(-|\d+\.)\s+", "", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            body.append(f"\\begin{{{kind}}}\n" + "\n".join(r"\item " + inline(t) for t in items) + f"\n\\end{{{kind}}}"); i = j; continue
        body.append(inline(ln) if ln.strip() else ""); i += 1
    tex = "\n".join([
        r"\documentclass[10pt]{article}", r"\usepackage[margin=1in]{geometry}", r"\usepackage{booktabs,graphicx,hyperref}",
        r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
        r"\title{" + inline(title) + "}", r"\author{NeoXider}", r"\date{v3.0, 2026-10-23}",
        r"\begin{document}", r"\maketitle", *body, r"\end{document}", ""])
    dst.write_text(tex, encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
