# VLAPB Overleaf Result Files

This folder contains figures, LaTeX tables, and full LaTeX documents for the VLAPB result section.

## Main files

- `vlapb_results_section.tex`: section-only file for importing into an existing paper.
- `vlapb_results_standalone.tex`: full compilable LaTeX document.
- `results/tab*.tex`: individual Overleaf-ready tables.
- `results/fig*.pdf`: vector figures for paper use.
- `results/fig*.png`: raster figures for slides or quick previews.

## In an existing Overleaf paper

Place the `results/` directory and `vlapb_results_section.tex` next to your main `.tex` file, then add:

```latex
\input{vlapb_results_section}
```

The main paper preamble should include:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{array}
```

## Standalone compile

Compile:

```latex
vlapb_results_standalone.tex
```

The current values are expected trend values for paper mockup and should be replaced with measured results after full VLAPB evaluation.
