import re
import os

def find_matching_brace(text, start_idx):
    count = 1
    i = start_idx + 1
    n = len(text)
    while i < n:
        if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
            count += 1
        elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
            count -= 1
            if count == 0:
                return i
        i += 1
    return -1

def get_next_arg(text, start_idx):
    i = start_idx
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text[i] == '%':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if text[i] == '{':
            end_idx = find_matching_brace(text, i)
            if end_idx != -1:
                return text[i+1:end_idx], end_idx + 1
            else:
                return None, -1
        break
    return None, -1

def convert_latex():
    input_path = os.path.join("article_final", "main.tex")
    output_path = os.path.join("jota_article", "main.tex")
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Preamble replacement
    # We will replace everything from the beginning of the file up to \begin{document}
    begin_doc_idx = text.find(r"\begin{document}")
    if begin_doc_idx == -1:
        print("Error: \\begin{document} not found")
        return
        
    preamble = r"""\documentclass[smallextended,referee,envcountsect]{svjour3}
\smartqed
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{etoolbox}
\AtBeginEnvironment{table}{\setlength{\extrarowheight}{2pt}}
\AtEndEnvironment{table}{\setlength{\extrarowheight}{1pt}}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{arydshln}
\usepackage[load-configurations=version-1]{siunitx}
\usepackage{mathtools}
\usepackage{bm}
\usepackage{subfigure}
\usepackage[numbers]{natbib}

\let\tens\undefined
\input{math_commands.tex}

% Norms and operators
\newcommand{\norm}[1]{\lVert #1\rVert}
\DeclarePairedDelimiter{\normf}{\|}{\|_\mathrm{F}}
\DeclarePairedDelimiter{\norms}{\|}{\|_{\mathrm{2}}}
\DeclarePairedDelimiter{\normc}{\|}{\|_{\mathrm{C}}}
\DeclarePairedDelimiter{\normn}{\|}{\|_{*}}
\DeclarePairedDelimiter{\normkfk}{\|}{\|_{\text{KF-}k}}
\DeclarePairedDelimiter{\normfstar}{\|}{\|_\mathrm{F*}}
\DeclarePairedDelimiter{\normftwo}{\|}{\|_\mathrm{F2}}
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}

% Inner product notation
\def\<#1,#2>{\langle #1,#2\rangle}

% Math operators
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\diag}{diag}

% Custom shortcuts
\renewcommand{\epsilon}{\varepsilon}
\newcommand{\Rmn}{\R^{m\times n}}

% Custom reference commands (not defined in svjour3)
\newcommand{\equationref}[1]{Equation~(\ref{#1})}
\newcommand{\sectionref}[1]{Section~\ref{#1}}
\newcommand{\lemmaref}[1]{Lemma~\ref{#1}}
\newcommand{\figureref}[1]{Figure~\ref{#1}}
\newcommand{\tableref}[1]{Table~\ref{#1}}
\newcommand{\corollaryref}[1]{Corollary~\ref{#1}}

\journalname{JOTA}

"""
    
    body = text[begin_doc_idx:]
    
    # 2. Reconstruct top matter (Title, Authors, Abstract, Keywords)
    # Let's define the JOTA title, subtitle (none), author, institute, date, and abstract.
    jota_title_block = r"""\title{The Ky Fan Norms and Beyond: Dual Norms and Combinations for Matrix Optimization}

\author{Alexey Kravatskiy \and Ivan Kozyrev \and Nikolai Kozlov \and Alexander Vinogradov \and Daniil Merkulov \and Ivan Oseledets}

\institute{Alexey Kravatskiy \at
             MIPT \\
             kravtskii.aiu@phystech.edu
           \and
             Ivan Kozyrev \at
             MIPT, INM RAS \\
             kozyrev.in@phystech.edu
           \and
             Nikolai Kozlov \at
             MIPT \\
             kozlov.na@phystech.edu
           \and
             Alexander Vinogradov \at
             MIPT \\
             vinogradov.am@phystech.edu
           \and
             Daniil Merkulov \at
             MIPT, Skoltech, HSE, AI4Science \\
             daniil.merkulov@phystech.edu
           \and
             Ivan Oseledets \at
             AIRI, Skoltech, INM RAS \\
             i.oseledets@skoltech.ru
}

\date{Received: date / Accepted: date}

\maketitle

"""
    
    # In body, let's find \begin{abstract} and \end{abstract}
    abs_start = body.find(r"\begin{abstract}")
    abs_end = body.find(r"\end{abstract}")
    if abs_start != -1 and abs_end != -1:
        abstract_content = body[abs_start:abs_end + len(r"\end{abstract}")]
        # We will insert keywords and subclass right after abstract
        keywords_block = abstract_content + "\n" + r"""\keywords{Matrix optimization \and Ky Fan norms \and Dual norms \and Linear minimization oracle \and Muon optimizer}
\subclass{90C30 \and 49M37 \and 68T07}

"""
        # Replace abstract in body with abstract + keywords
        body = body[:abs_start] + keywords_block + body[abs_end + len(r"\end{abstract}"):]
        
    # We replace everything from \begin{document} up to the start of Introduction with \begin{document} + jota_title_block + abstract + keywords
    intro_start = body.find(r"\section{Introduction}")
    if intro_start == -1:
        print("Error: \\section{Introduction} not found")
        return
        
    # Extract the abstract/keywords part we just modified
    modified_abs_keywords = body[:intro_start]
    # Replace it with \begin{document} + title block + abstract/keywords (which is already in modified_abs_keywords, minus the raw PMLR title/author info)
    abs_start_mod = modified_abs_keywords.find(r"\begin{abstract}")
    if abs_start_mod != -1:
        body = "\\begin{document}\n\n" + jota_title_block + modified_abs_keywords[abs_start_mod:] + body[intro_start:]
    else:
        body = "\\begin{document}\n\n" + jota_title_block + body[intro_start:]

    # 3. Parse \floatconts
    idx = 0
    while True:
        match_idx = body.find(r'\floatconts', idx)
        if match_idx == -1:
            break
        
        arg1, end_idx1 = get_next_arg(body, match_idx + len(r'\floatconts'))
        if arg1 is None:
            idx = match_idx + 1
            continue
        arg2, end_idx2 = get_next_arg(body, end_idx1)
        if arg2 is None:
            idx = match_idx + 1
            continue
        arg3, end_idx3 = get_next_arg(body, end_idx2)
        if arg3 is None:
            idx = match_idx + 1
            continue
        
        # Scan backward to determine if we are inside figure or table
        bg_table = body.rfind(r'\begin{table}', 0, match_idx)
        bg_table_star = body.rfind(r'\begin{table*}', 0, match_idx)
        bg_fig = body.rfind(r'\begin{figure}', 0, match_idx)
        bg_fig_star = body.rfind(r'\begin{figure*}', 0, match_idx)
        
        max_bg = max(bg_table, bg_table_star, bg_fig, bg_fig_star)
        is_table = (max_bg == bg_table or max_bg == bg_table_star)
        
        # Format the caption with label
        caption_with_label = arg2
        if '\\label{' not in caption_with_label:
            cap_match = re.search(r'\\caption\s*\{', caption_with_label)
            if cap_match:
                cap_start = cap_match.end() - 1
                cap_end = find_matching_brace(caption_with_label, cap_start)
                if cap_end != -1:
                    caption_with_label = caption_with_label[:cap_end+1] + f"\\label{{{arg1}}}" + caption_with_label[cap_end+1:]
                else:
                    caption_with_label = caption_with_label + f"\\label{{{arg1}}}"
            else:
                caption_with_label = caption_with_label + f"\\label{{{arg1}}}"
        
        if is_table:
            replacement = f"\n{caption_with_label}\n{arg3}\n"
        else:
            replacement = f"\n{arg3}\n{caption_with_label}\n"
            
        body = body[:match_idx] + replacement + body[end_idx3:]
        idx = match_idx + len(replacement)

    # 4. Handle acknowledgements block
    # Convert \section*{Acknowledgements} ... to \begin{acknowledgements} ... \end{acknowledgements}
    ack_section = r"\section*{Acknowledgements}"
    ack_idx = body.find(ack_section)
    if ack_idx != -1:
        # Find the next section or bibliography
        next_idx = body.find(r"\bibliography", ack_idx)
        if next_idx == -1:
            next_idx = body.find(r"\appendix", ack_idx)
        if next_idx != -1:
            ack_text = body[ack_idx + len(ack_section):next_idx].strip()
            new_ack = f"\\begin{{acknowledgements}}\n{ack_text}\n\\end{{acknowledgements}}\n\n"
            body = body[:ack_idx] + new_ack + body[next_idx:]

    # 5. Convert bibliography command
    # Replace \bibliography{icomp2024_conference} with:
    # \bibliographystyle{spmpsci}
    # \bibliography{icomp2024_conference}
    body = body.replace(r"\bibliography{icomp2024_conference}", r"\bibliographystyle{spmpsci}" + "\n" + r"\bibliography{icomp2024_conference}")

    # 6. Add QED to proofs (using lambda to avoid backslash interpretation issues in re.sub)
    body = re.sub(r'which completes the proof\.\s*\\end\{proof\}', lambda m: 'which completes the proof.\\qed\n\\end{proof}', body)
    body = re.sub(r'completing the proof\.\s*\\end\{proof\}', lambda m: 'completing the proof.\\qed\n\\end{proof}', body)
    body = re.sub(r'which proves the lemma\.\s*\\end\{proof\}', lambda m: 'which proves the lemma.\\qed\n\\end{proof}', body)

    # 7. Convert \citet to manual author + \cite
    body = body.replace(r"\citet{cesista2025schattenp}", r"Cesista~\cite{cesista2025schattenp}")
    body = body.replace(r"\citet{liu2025muon}", r"Liu~et~al.~\cite{liu2025muon}")
    body = body.replace(r"\citet{shah2025practical}", r"Shah~et~al.~\cite{shah2025practical}")
    body = body.replace(r"\citet{chen2025cosmoshybridadaptive}", r"Chen~et~al.~\cite{chen2025cosmoshybridadaptive}")
    body = body.replace(r"\citet{amsel2025polar}", r"Amsel~et~al.~\cite{amsel2025polar}")
    body = body.replace(r"\citet{grishina2025chebyshev}", r"Grishina~et~al.~\cite{grishina2025chebyshev}")
    body = body.replace(r"\citet{si2025adamuon}", r"Si~et~al.~\cite{si2025adamuon}")
    body = body.replace(r"\citet{veprikov2025preconditioned}", r"Veprikov~et~al.~\cite{veprikov2025preconditioned}")
    body = body.replace(r"\citet{li2025normuon}", r"Li~et~al.~\cite{li2025normuon}")
    body = body.replace(r"\citet{yu2012arithmetic}", r"Yu~\cite{yu2012arithmetic}")
    body = body.replace(r"\citet{kovalev2025understanding}", r"Kovalev~\cite{kovalev2025understanding}")

    # Combine preamble and body
    final_text = preamble + body
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print("Conversion complete! jota_article/main.tex written.")

if __name__ == "__main__":
    convert_latex()
