\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tex-notes}[2026/05/13 TeX Notes class]

%% Base class: single-column 11pt research notes / companion documents.
\LoadClass[11pt,a4paper]{article}

%% --- Page geometry -------------------------------------------------------
\RequirePackage[margin=2cm]{geometry}

%% --- Fonts & micro-typography -------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}

%% --- Standard helpers ---------------------------------------------------
\RequirePackage{booktabs}
\RequirePackage{enumitem}
\RequirePackage[hidelinks]{hyperref}

%% --- Project shared packages -------------------------------------------
%% tex-colors  : palette (stateFill/Edge, opFill/Edge, ioFill/Edge, arrowCol)
%% tex-math    : amsmath/amssymb/amsthm + \ConvT, \BN, \ReLU, ...
%% tex-tikz    : tikz + libraries + state/iostate/op/arr/stagelabel styles
\RequirePackage{tex-colors}
\RequirePackage{tex-math}
\RequirePackage{tex-tikz}

%% --- Section title styling ---------------------------------------------
\RequirePackage{titlesec}
\titleformat{\section}
  {\normalfont\large\bfseries\color{stateEdge}}{\thesection}{0.6em}{}
\titleformat{\subsection}
  {\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}

%% --- Paragraph spacing --------------------------------------------------
\setlength{\parskip}{0.5ex plus 0.2ex}
\setlength{\parindent}{0pt}

%% --- Block-header macro -------------------------------------------------
%%   \blockhead{<palette-color>}{<title text>}
%%   Used by content/walkthrough/*.tex to introduce each block. The
%%   palette colour is one of: ioEdge, opEdge, stateEdge.
\newcommand{\blockhead}[2]{%
  \par\medskip
  \noindent\colorbox{#1!18}{%
    \rule[-2pt]{0pt}{12pt}%
    \hspace{4pt}\textbf{#2}\hspace{4pt}%
  }\par\smallskip
}

\endinput
