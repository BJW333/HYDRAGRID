#!/bin/bash
# ============================================================
# HYDRAGRID — One-Command Setup
# ============================================================
# Run this on your VPS or local machine:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=================================================="
echo "  HYDRAGRID — Automated Setup"
echo "=================================================="
echo ""

# Check Python version
echo -n "Checking Python... "
if command -v python3.10 &> /dev/null; then
    PYTHON=python3.10
    echo -e "${GREEN}python3.10 found${NC}"
elif command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PY_VER | cut -d'.' -f1)
    MINOR=$(echo $PY_VER | cut -d'.' -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
        PYTHON=python3
        echo -e "${GREEN}python3 ($PY_VER) found${NC}"
    else
        echo -e "${RED}Python 3.10+ required (found $PY_VER)${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python 3 not found. Install Python 3.10+${NC}"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
$PYTHON -m pip install --upgrade pip --quiet 2>/dev/null || true
$PYTHON -m pip install -r requirements.txt --quiet --break-system-packages 2>/dev/null || \
$PYTHON -m pip install -r requirements.txt --quiet 2>/dev/null || {
    echo -e "${RED}Failed to install dependencies${NC}"
    echo "Try manually: $PYTHON -m pip install -r requirements.txt"
    exit 1
}
echo -e "${GREEN}Dependencies installed${NC}"

# Create directories
echo ""
echo "Setting up directories..."
mkdir -p data/historical
mkdir -p logs
echo -e "${GREEN}Directories created${NC}"

# Check API keys
echo ""
echo "Checking API keys..."
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
    echo -e "${YELLOW}⚠️  API keys not set.${NC}"
    echo ""
    read -p "Enter your Alpaca API Key: " API_KEY
    read -p "Enter your Alpaca API Secret: " API_SECRET
    
    if [ -n "$API_KEY" ] && [ -n "$API_SECRET" ]; then
        echo "" >> ~/.bashrc
        echo "# HYDRAGRID API Keys" >> ~/.bashrc
        echo "export ALPACA_API_KEY=\"$API_KEY\"" >> ~/.bashrc
        echo "export ALPACA_API_SECRET=\"$API_SECRET\"" >> ~/.bashrc
        export ALPACA_API_KEY="$API_KEY"
        export ALPACA_API_SECRET="$API_SECRET"
        echo -e "${GREEN}API keys saved to ~/.bashrc${NC}"
    else
        echo -e "${YELLOW}Skipped. Set them later:${NC}"
        echo '  export ALPACA_API_KEY="your-key"'
        echo '  export ALPACA_API_SECRET="your-secret"'
    fi
else
    echo -e "${GREEN}API keys found${NC}"
fi

# Set capital
echo ""
CURRENT_CAPITAL=$(grep "STARTING_CAPITAL" config/settings.py | head -1 | grep -oP '\d+')
echo "Current starting capital: \$$CURRENT_CAPITAL"
read -p "Set starting capital (press Enter to keep \$$CURRENT_CAPITAL): " NEW_CAPITAL

if [ -n "$NEW_CAPITAL" ]; then
    sed -i "s/STARTING_CAPITAL = $CURRENT_CAPITAL/STARTING_CAPITAL = $NEW_CAPITAL/" config/settings.py 2>/dev/null || \
    sed -i '' "s/STARTING_CAPITAL = $CURRENT_CAPITAL/STARTING_CAPITAL = $NEW_CAPITAL/" config/settings.py
    echo -e "${GREEN}Capital set to \$$NEW_CAPITAL${NC}"
fi

# Test connection
echo ""
echo "Testing exchange connection..."
if [ -n "$ALPACA_API_KEY" ] && [ -n "$ALPACA_API_SECRET" ]; then
    $PYTHON test_alpaca.py 2>/dev/null && echo -e "${GREEN}Connection successful${NC}" || {
        echo -e "${YELLOW}Connection test failed — check your API keys${NC}"
    }
else
    echo -e "${YELLOW}Skipped — API keys not set${NC}"
fi

# Download historical data for backtesting
echo ""
read -p "Download historical data for backtesting? (y/n): " DOWNLOAD
if [ "$DOWNLOAD" = "y" ] || [ "$DOWNLOAD" = "Y" ]; then
    echo "Downloading 1 year of data for SOL, ETH, LINK, AVAX..."
    $PYTHON download_data.py --symbols SOL ETH LINK AVAX --days 365
    echo -e "${GREEN}Data downloaded${NC}"
fi

# Summary
echo ""
echo "=================================================="
echo -e "  ${GREEN}HYDRAGRID Setup Complete${NC}"
echo "=================================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Paper trading (test mode — start here):"
echo "     $PYTHON live_trader.py"
echo ""
echo "  2. Run backtest:"
echo "     $PYTHON realistic_backtest.py"
echo ""
echo "  3. Go live (after paper testing):"
echo "     $PYTHON live_trader.py --live"
echo ""
echo "  4. Run in background (VPS):"
echo "     nohup $PYTHON live_trader.py > ~/hydragrid_run.log 2>&1 &"
echo ""
echo "  5. Check status:"
echo "     tail -50 ~/hydragrid_run.log"
echo ""
echo "  See SETUP.md for full deployment guide."
echo "  See PERFORMANCE.md for backtest results."
echo "=================================================="
