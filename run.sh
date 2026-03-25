#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NiftySignals Pro — One-Click Setup & Launch
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║   NiftySignals Pro — F&O Trading Signal System      ║"
echo "║   Setting up your trading dashboard...               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "🚀 Launching NiftySignals Pro..."
echo "   Dashboard will open at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit (FIXED FOR RAILWAY)
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#00e676" \
    --theme.backgroundColor "#0e1117" \
    --theme.secondaryBackgroundColor "#1a1a2e" \
    --theme.textColor "#ffffff"
