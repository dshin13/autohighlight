#!/bin/bash
# Setup script for AutoHighlight autohighlights branch
# This script prepares AutoHighlight for integration with BallQ_beta

set -e

echo "========================================="
echo "AutoHighlight - BallQ_beta Setup Script"
echo "========================================="
echo ""

# Get the directory this script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "1. Creating autohighlights branch..."
if git rev-parse --verify autohighlights > /dev/null 2>&1; then
    echo "   Branch 'autohighlights' already exists. Switching to it..."
    git checkout autohighlights
else
    echo "   Creating new branch 'autohighlights'..."
    git checkout -b autohighlights
fi

echo ""
echo "2. Staging FastAPI integration files..."
git add fastapi_wrapper.py ballq_api.py setup.py BALLQ_INTEGRATION.md README.md

echo ""
echo "3. Checking git status..."
git status

echo ""
echo "4. Committing changes..."
git commit -m "Add FastAPI integration for BallQ_beta

- Added fastapi_wrapper.py: Core API wrapper for AutoHighlight
- Added ballq_api.py: FastAPI router with endpoints for BallQ_beta integration
- Added setup.py: Configuration module for easy initialization
- Added BALLQ_INTEGRATION.md: Complete integration documentation
- Updated README.md: Added BallQ_beta integration section

This branch is designed to work as a submodule in the BallQ_beta project."

echo ""
echo "5. Branch information:"
git log --oneline -3

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Push this branch to GitHub:"
echo "   git push -u origin autohighlights"
echo ""
echo "2. In BallQ_beta repository, add AutoHighlight as a submodule:"
echo "   git submodule add -b autohighlights https://github.com/dannywhite908/autohighlight.git services/autohighlight"
echo ""
echo "3. See BALLQ_INTEGRATION.md for detailed integration instructions"
echo ""
