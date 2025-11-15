#!/bin/bash
# Enable GitHub Pages via API after repo is public
# Copyright (C) 2025, Shyamal Suhana Chandra

REPO="Sapana-Micro-Software/ddrkam"

echo "🔧 Enabling GitHub Pages for $REPO..."

# Method 1: Try to enable via API with workflow build type
echo "Attempting to enable Pages with GitHub Actions workflow..."
gh api -X POST "/repos/$REPO/pages" \
  --field 'build_type=workflow' 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Pages enabled successfully with workflow build type!"
    exit 0
fi

# Method 2: Try with source branch
echo "Attempting to enable Pages with branch source..."
gh api -X POST "/repos/$REPO/pages" \
  --field 'source[branch]=main' \
  --field 'source[path]=/docs' 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Pages enabled successfully with branch source!"
    exit 0
fi

# Method 3: Manual instructions
echo ""
echo "⚠️  Automatic enablement via API failed."
echo "Please enable manually:"
echo ""
echo "1. Visit: https://github.com/$REPO/settings/pages"
echo "2. Under 'Source', select: GitHub Actions"
echo "3. Click Save"
echo ""
echo "Or use branch deployment:"
echo "1. Visit: https://github.com/$REPO/settings/pages"
echo "2. Select 'Deploy from a branch'"
echo "3. Branch: main, Folder: /docs"
echo "4. Click Save"
