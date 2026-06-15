# HYDRAGRID — Setup Guide

Complete step-by-step guide from zero to running bot.

---

## Step 1: Create Alpaca Account (5 minutes)

1. Go to [app.alpaca.markets](https://app.alpaca.markets)
2. Sign up for a free account
3. You'll start in **Paper Trading** mode (no real money)
4. Navigate to **API** in the left sidebar
5. Click **Generate** to create API keys
6. Save both the **Key** and **Secret** — you'll need them below

**For live trading (real money):**
- Click "Open Live Account"
- Tax ID type = SSN (your Social Security Number)
- Approval takes ~1 business day
- Fund via bank transfer ($500 minimum recommended)
- Generate separate **Live** API keys

---

## Step 2: Set Up Your Server (15 minutes)

HYDRAGRID needs to run 24/7. A cloud VPS is recommended.

### Option A: Oracle Cloud (Free Forever)

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com)
2. Create a **Compute Instance**:
   - Shape: VM.Standard.A1.Flex (ARM, free tier)
   - OS: Ubuntu 24.04
   - 1 CPU, 6GB RAM is plenty
3. Download the SSH key during setup
4. Note the public IP address

### Option B: Any Linux VPS

Any VPS with Python 3.10+ works:
- DigitalOcean ($6/month)
- Linode ($5/month)
- AWS Lightsail ($3.50/month)

---

## Step 3: Deploy HYDRAGRID (5 minutes)

### Upload the bot to your server:

```bash
# From your local machine:
scp -r -i /path/to/your-ssh-key.key HYDRAGRID/ ubuntu@YOUR_SERVER_IP:~/hydragrid/
```

### SSH into your server:

```bash
ssh -i /path/to/your-ssh-key.key ubuntu@YOUR_SERVER_IP
```

### Run the setup script:

```bash
cd ~/hydragrid
chmod +x setup.sh
./setup.sh
```

This installs all dependencies and configures the environment.

---

## Step 4: Set API Keys

```bash
# Add to your shell profile:
echo 'export ALPACA_API_KEY="your-key-here"' >> ~/.bashrc
echo 'export ALPACA_API_SECRET="your-secret-here"' >> ~/.bashrc
source ~/.bashrc
```

**Important:** Use your **Paper** keys first. Switch to **Live** keys only after confirming the bot works in paper mode.

---

## Step 5: Configure Capital

Edit the starting capital in `config/settings.py`:

```bash
nano ~/hydragrid/config/settings.py
```

Find this line and set your capital:
```python
STARTING_CAPITAL = 1000  # Set to your actual deposit amount
```

Save and exit (Ctrl+X, Y, Enter).

---

## Step 6: Start the Bot

### Paper trading (test mode — start here):

```bash
cd ~/hydragrid
nohup python3 live_trader.py > ~/hydragrid_run.log 2>&1 &
```

### Live trading (real money):

```bash
cd ~/hydragrid
nohup python3 live_trader.py --live > ~/hydragrid_run.log 2>&1 &
```

### Verify it's running:

```bash
# Check the log
tail -50 ~/hydragrid_run.log

# Check the process
ps aux | grep live_trader
```

You should see grid setup messages for SOL, ETH, LINK, and AVAX.

---

## Step 7: Monitor

### Check status anytime:

```bash
tail -100 ~/hydragrid_run.log
```

### Check Alpaca dashboard:

Go to [app.alpaca.markets](https://app.alpaca.markets) to see your portfolio value, open positions, and trade history.

### Set up Telegram alerts (optional):

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add to your environment:

```bash
echo 'export TELEGRAM_BOT_TOKEN="your-bot-token"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your-chat-id"' >> ~/.bashrc
source ~/.bashrc
```

Restart the bot to activate alerts.

---

## Step 8: Going Live

**Pre-flight checklist:**

- [ ] Bot has run in paper mode for at least 7 days
- [ ] Paper results show positive P&L
- [ ] Alpaca live account is approved and funded
- [ ] Live API keys are generated
- [ ] Capital amount set correctly in settings.py

**Switch to live:**

```bash
# Stop paper bot
pkill -f live_trader

# Update to live API keys
nano ~/.bashrc
# Change ALPACA_API_KEY and ALPACA_API_SECRET to live keys
source ~/.bashrc

# Start live
cd ~/hydragrid
nohup python3 live_trader.py --live > ~/hydragrid_run.log 2>&1 &
```

---

## Common Commands

| Command | What it does |
|---|---|
| `tail -50 ~/hydragrid_run.log` | Check recent bot activity |
| `pkill -f live_trader` | Stop the bot |
| `ps aux \| grep live_trader` | Check if bot is running |
| `python3 live_trader.py --test` | Run one cycle and exit |
| `python3 backtest.py` | Run historical backtest |
| `python3 realistic_backtest.py` | Full friction analysis |
| `python3 grid_sweep.py` | Parameter optimization |

---

## Troubleshooting

**Bot won't start — "API key and secret required"**
→ Make sure you exported your keys: `echo $ALPACA_API_KEY`

**403 Forbidden on orders**
→ Crypto trading may not be enabled. Check Alpaca dashboard → Account → ensure crypto is enabled.

**Bot stops after a few hours**
→ Check `~/hydragrid_run.log` for errors. Common cause: API rate limiting. The bot handles this automatically but extreme cases may need the check interval increased (`--interval 120`).

**Orders not filling**
→ Normal during low-volatility periods. The grid waits for price to move to its levels. Check the log — you should see "check_fills" running every 60 seconds.

---

## Disclaimer

Trading cryptocurrency involves significant risk. Past performance (backtested or paper traded) does not guarantee future results. Only trade with money you can afford to lose. This software is provided as-is with no warranty.
