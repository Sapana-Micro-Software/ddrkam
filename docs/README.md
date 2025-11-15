# DDRKAM Documentation Site

This directory contains the GitHub Pages site for DDRKAM.

## Building Locally

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then visit `http://localhost:4000/ddrkam`

## PDF Generation

To generate PDFs from LaTeX sources:

```bash
cd docs
pdflatex paper.tex
pdflatex presentation.tex
pdflatex reference_manual.tex
```

Place the generated PDFs in the `docs/` directory for the site to link to them.
