# GitHub Pages Setup Instructions

The GitHub Pages site has been created and pushed to the repository. The workflow includes automatic enablement, but if you encounter issues:

## Option 1: Automatic (Recommended)

The GitHub Actions workflow will automatically enable and deploy Pages. Just push to the main branch and the workflow will run.

## Option 2: Manual Enablement

If automatic enablement fails, manually enable Pages:

1. Go to your repository: https://github.com/Sapana-Micro-Software/ddrkam
2. Click on **Settings** (top menu)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Source**: `GitHub Actions` (for workflow-based deployment)
   - OR `Deploy from a branch` → `main` → `/docs` (for static deployment)
5. Click **Save**

The site will be available at: **https://sapana-micro-software.github.io/ddrkam**

## Troubleshooting

If you see "Pages site failed" error:
1. Check that the repository has Pages enabled in Settings → Pages
2. Ensure the workflow has proper permissions (should be automatic)
3. The workflow includes `enablement: true` parameter to auto-enable Pages
4. If issues persist, manually enable Pages using Option 2 above

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
