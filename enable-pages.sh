#!/bin/bash
# Script to enable GitHub Pages via API
# Copyright (C) 2025, Shyamal Suhana Chandra

echo "Attempting to enable GitHub Pages..."

# Try to enable Pages via API
gh api -X POST /repos/Sapana-Micro-Software/ddrkam/pages \
  -f '{"source":{"branch":"main","path":"/docs"}}' 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Pages enabled successfully!"
    echo "Site will be available at: https://sapana-micro-software.github.io/ddrkam"
else
    echo ""
    echo "⚠️  Automatic enablement failed. Please enable manually:"
    echo ""
    echo "1. Go to: https://github.com/Sapana-Micro-Software/ddrkam/settings/pages"
    echo "2. Under 'Source', select: GitHub Actions"
    echo "3. Click Save"
    echo ""
    echo "The workflow will then deploy automatically."
fi
