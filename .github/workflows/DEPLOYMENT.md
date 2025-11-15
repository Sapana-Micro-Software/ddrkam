# GitHub Pages Deployment Guide

## Overview

This repository uses GitHub Actions to automatically deploy GitHub Pages whenever changes are pushed to the `main` branch.

## Workflows

### 1. `pages.yml` - Jekyll-based Deployment

**Primary workflow** that builds the site using Jekyll and deploys to GitHub Pages.

**Features:**
- Builds site with Jekyll
- Handles Jekyll build failures gracefully
- Falls back to static files if Jekyll fails
- Verifies build output
- Automatic deployment on push to `main` branch
- Manual workflow dispatch support

**Triggers:**
- Push to `main` branch (when `docs/**` files change)
- Manual workflow dispatch
- Pull requests to `main` branch

### 2. `pages-static.yml` - Static File Deployment

**Alternative workflow** for direct static file deployment (no Jekyll).

**Features:**
- Direct static file deployment (faster)
- Comprehensive file verification
- No Jekyll dependencies
- Deployment summary with URL

**Use when:**
- Jekyll build fails
- You need faster deployment
- You want to deploy static files only

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to: `https://github.com/Sapana-Micro-Software/ddrkam/settings/pages`
2. Under "Source", select **"GitHub Actions"**
3. Save the settings

### 2. Verify Workflow Permissions

The workflows require the following permissions:
- `contents: read` - Read repository contents
- `pages: write` - Write to GitHub Pages
- `id-token: write` - For OIDC authentication

These are automatically configured in the workflow files.

### 3. Manual Deployment

To manually trigger a deployment:

1. Go to: `https://github.com/Sapana-Micro-Software/ddrkam/actions`
2. Select "Deploy GitHub Pages" or "Deploy GitHub Pages (Static)"
3. Click "Run workflow"
4. Select the branch (usually `main`)
5. Click "Run workflow"

## What Gets Deployed

The workflows deploy all files from the `docs/` directory:

- **HTML files**: `index.html` and all HTML files
- **JavaScript files**: All files in `assets/js/` (charts.js, main.js, test-visualizations.js, etc.)
- **CSS files**: All files in `assets/css/` (main.css)
- **PDF files**: `paper.pdf`, `presentation.pdf`, `reference_manual.pdf`
- **SVG assets**: All files in `assets/svg/`
- **Documentation**: All markdown files (BENCHMARKS.md, COMPARISON.md, etc.)
- **Configuration**: `_config.yml`, `sitemap.xml`, `robots.txt`

## Deployment Status

### View Deployment Status

1. Go to: `https://github.com/Sapana-Micro-Software/ddrkam/actions`
2. Click on the latest workflow run
3. View the build and deploy steps

### Deployment URL

After successful deployment, the site will be available at:
- **Production**: `https://sapana-micro-software.github.io/ddrkam/`

### Check Deployment

1. Wait for the workflow to complete (usually 1-2 minutes)
2. Visit the deployment URL
3. Verify all files are accessible

## Troubleshooting

### Workflow Fails

1. **Check workflow logs**: Go to Actions tab and view the failed run
2. **Common issues**:
   - Jekyll build errors: Check `_config.yml` and Jekyll dependencies
   - Missing files: Verify all required files are in `docs/` directory
   - Permission errors: Ensure GitHub Pages is enabled and using GitHub Actions

### Jekyll Build Fails

The `pages.yml` workflow automatically falls back to static files if Jekyll fails. Alternatively:
1. Use `pages-static.yml` workflow for direct static deployment
2. Check Jekyll configuration in `_config.yml`
3. Verify `Gemfile` has correct dependencies

### Files Not Updating

1. **Clear browser cache**: Hard refresh (Ctrl+F5 or Cmd+Shift+R)
2. **Check file paths**: Ensure file paths in HTML are correct
3. **Verify deployment**: Check Actions tab to confirm deployment completed
4. **Wait for propagation**: GitHub Pages updates may take a few minutes

### Manual Fix

If automatic deployment fails:
1. Use "Deploy GitHub Pages (Static)" workflow
2. Manually verify all files are in `docs/` directory
3. Check file permissions and paths

## Best Practices

1. **Test locally**: Test Jekyll build locally before pushing:
   ```bash
   cd docs
   bundle install
   bundle exec jekyll build
   ```

2. **Verify changes**: Always check the Actions tab after pushing
3. **Monitor deployments**: Set up notifications for failed deployments
4. **Keep workflows updated**: Use latest action versions

## Support

For issues or questions:
- Check workflow logs in the Actions tab
- Review GitHub Pages documentation
- Contact: sapanamicrosoftware@duck.com

---

**Copyright © 2025, Shyamal Suhana Chandra**
