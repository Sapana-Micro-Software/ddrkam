# GitHub Pages Setup Instructions

The GitHub Pages site has been created and pushed to the repository. To enable it:

## Steps to Enable GitHub Pages

1. Go to your repository: https://github.com/Sapana-Micro-Software/ddrkam
2. Click on **Settings** (top menu)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main`
   - **Folder**: `/docs`
5. Click **Save**

The site will be available at: **https://sapana-micro-software.github.io/ddrkam**

## What's Included

- ✅ Modern, responsive HTML/CSS/JavaScript website
- ✅ Interactive benchmark visualizations with charts
- ✅ Performance statistics (accuracy, speed, memory, convergence)
- ✅ Links to paper, presentation, and reference manual (LaTeX sources)
- ✅ Complete references and citations
- ✅ Dedication to University of Washington Coursera course
- ✅ Interactive ODE visualization (Lorenz attractor)
- ✅ Smooth animations and transitions
- ✅ Mobile-responsive design

## Local Development

To test locally:

```bash
cd docs
bundle install
bundle exec jekyll serve --baseurl /ddrkam
```

Visit: http://localhost:4000/ddrkam
