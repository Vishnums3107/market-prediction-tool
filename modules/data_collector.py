import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import ta
import requests
import json
import time
import os
import joblib
from typing import Dict, List, Union, Optional
import talib
class DataCollector:
    def __init__(self, alpha_vantage_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.newsapi_key = newsapi_key
        self.cache_dir = 'data'
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{interval}.pkl")
    def _generate_dummy_data(self):
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
        prices = np.cumprod(1 + np.random.normal(0.001, 0.01, 100)) * 100
        return pd.DataFrame({
            'Open': prices,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices * 1.002,
            'Volume': np.random.randint(10000, 50000, 100)
        }, index=dates)     
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required technical indicators"""
        df = df.copy()
        
        # Calculate indicators (using lowercase column names)
        if 'Close' in df.columns:
            df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist']= talib.MACD(df['Close'])
            upper, middle, lower = talib.BBANDS(df['Close'])
            df['bollinger_upper'] = upper
            df['bollinger_lower'] = lower
            
        return df
    def fetch_price_data(self, symbol: str, interval: str = '15m', period: str = '7d'):
        try:
            # Add forex suffix if missing
            if "=X" not in symbol and "-" not in symbol and "^" not in symbol:
                symbol = f"{symbol}.NS"  # Default to NSE if no suffix
            
            # Special handling for crypto
            if "USD" in symbol and "-" not in symbol:
                symbol = symbol.replace("USD", "-USD")
            try:
                rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
                df['rsi'] = rsi_indicator.rsi()
                print("RSI column added successfully")
            except Exception as e:
                print("Failed to compute RSI:", e)
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=True
            )
            
            if df.empty:
                return df
            df = self._add_technical_indicators(df)    
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"Data fetch error: {e}")
            return self._generate_dummy_data()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        return df

    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        if not self.newsapi_key:
            return []
            
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # This is a placeholder - in practice you'd use the NewsAPI client
            # response = requests.get(f"https://newsapi.org/v2/everything?q={symbol}...")
            # return response.json().get('articles', [])
            
            return [{
                'title': f"Sample news about {symbol}",
                'description': "This is a placeholder news item",
                'publishedAt': datetime.datetime.now().isoformat()
            }]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []