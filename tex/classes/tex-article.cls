\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tex-article}[2026/05/13 TeX Article class]
\LoadClass[10pt,twocolumn,a4paper]{article}
\RequirePackage[margin=2cm,columnsep=0.7cm]{geometry}
\RequirePackage[T1]{fontenc}\RequirePackage{lmodern}\RequirePackage{microtype}
\RequirePackage{amsmath,amssymb,amsthm,booktabs,enumitem,authblk}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{tex-colors}\RequirePackage{tex-math}
\RequirePackage{tex-tikz}\RequirePackage{tex-tables}
\RequirePackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
