"""
Alerts Module
=============
Send notifications via Telegram, Discord, or console.
"""

import requests
import json
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    ENABLE_NOTIFICATIONS, VERBOSE
)


class AlertManager:
    """
    Handles all notifications and alerts.
    
    Supports:
    - Telegram
    - Discord webhooks
    - Console output
    - Log files
    """
    
    def __init__(self, 
                 telegram_token: str = TELEGRAM_BOT_TOKEN,
                 telegram_chat_id: str = TELEGRAM_CHAT_ID,
                 discord_webhook: str = "",
                 enable_telegram: bool = True,
                 enable_discord: bool = False,
                 enable_console: bool = True,
                 log_file: str = "logs/alerts.log"):
        
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook
        self.enable_telegram = enable_telegram and telegram_token and telegram_chat_id
        self.enable_discord = enable_discord and discord_webhook
        self.enable_console = enable_console
        
        # Setup log file
        self.log_file = Path(__file__).parent.parent / log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _log_to_file(self, message: str):
        """Write message to log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram."""
        if not self.enable_telegram:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def _send_discord(self, message: str) -> bool:
        """Send message via Discord webhook."""
        if not self.enable_discord:
            return False
        
        try:
            payload = {'content': message}
            response = requests.post(
                self.discord_webhook, 
                json=payload, 
                timeout=10
            )
            return response.status_code in [200, 204]
        except Exception as e:
            print(f"Discord error: {e}")
            return False
    
    def send(self, message: str, level: str = "INFO"):
        """
        Send alert through all enabled channels.
        
        Args:
            message: Alert message
            level: INFO, WARNING, SIGNAL, TRADE, ERROR
        """
        # Format with level
        formatted = f"[{level}] {message}"
        
        # Console
        if self.enable_console:
            if level == "ERROR":
                print(f"❌ {message}")
            elif level == "WARNING":
                print(f"⚠️  {message}")
            elif level == "SIGNAL":
                print(f"🎯 {message}")
            elif level == "TRADE":
                print(f"💰 {message}")
            else:
                print(f"ℹ️  {message}")
        
        # Log file
        self._log_to_file(formatted)
        
        # Telegram (only for important messages)
        if level in ["SIGNAL", "TRADE", "ERROR", "WARNING"]:
            self._send_telegram(message)
        
        # Discord
        if level in ["SIGNAL", "TRADE", "ERROR"]:
            self._send_discord(message)
    
    def signal_alert(self, symbol: str, direction: str, price: float, 
                     stop_loss: float, take_profit: float):
        """Send a trading signal alert."""
        emoji = "🟢" if direction == "LONG" else "🔴"
        message = f"""
{emoji} <b>SIGNAL: {direction} {symbol}</b>

📊 Entry Price: ${price:,.2f}
🛑 Stop Loss: ${stop_loss:,.2f}
🎯 Take Profit: ${take_profit:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message.strip(), level="SIGNAL")
    
    def trade_alert(self, symbol: str, action: str, direction: str,
                    price: float, size: float, pnl: Optional[float] = None):
        """Send a trade execution alert."""
        if action == "OPEN":
            emoji = "🟢" if direction == "LONG" else "🔴"
            message = f"""
{emoji} <b>TRADE OPENED: {direction} {symbol}</b>

📊 Entry: ${price:,.2f}
📦 Size: {size:.4f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:  # CLOSE
            pnl_emoji = "✅" if pnl and pnl > 0 else "❌"
            pnl_str = f"${pnl:,.2f}" if pnl else "N/A"
            message = f"""
{pnl_emoji} <b>TRADE CLOSED: {symbol}</b>

📊 Exit: ${price:,.2f}
💰 P&L: {pnl_str}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message.strip(), level="TRADE")
    
    def error_alert(self, error: str):
        """Send an error alert."""
        message = f"🚨 <b>ERROR</b>\n\n{error}"
        self.send(message, level="ERROR")
    
    def status_alert(self, message: str):
        """Send a status update."""
        self.send(message, level="WARNING")  # So it goes to Telegram


# Test function
if __name__ == "__main__":
    alerts = AlertManager(enable_console=True)
    
    print("Testing alert system...\n")
    
    alerts.status_alert("Bot started successfully")
    alerts.signal_alert("LINK/USD", "LONG", 15.50, 14.80, 17.20)
    alerts.trade_alert("LINK/USD", "OPEN", "LONG", 15.52, 64.5)
    alerts.trade_alert("LINK/USD", "CLOSE", "LONG", 17.15, 64.5, pnl=105.23)
    alerts.error_alert("Connection timeout - retrying...")
    
    print("\n✅ Alert test complete")
