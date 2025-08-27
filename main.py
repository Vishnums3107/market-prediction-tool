import streamlit as st
from modules.data_collector import DataCollector
from modules.sentiment import SentimentAnalyzer
from modules.predictor import MarketPredictor
from modules.dashboard import TradingDashboard
import configparser
import os

def load_config():
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    return config

def initialize_components(config):
    data_collector = DataCollector(
        alpha_vantage_key=config['api_keys'].get('alpha_vantage'),
        newsapi_key=config['api_keys'].get('newsapi')
    )
    sentiment_analyzer = SentimentAnalyzer()
    market_predictor = MarketPredictor(
        model_dir='saved_models',
        lookback=int(config['model'].get('lookback_window', 60)),
        epochs=int(config['model'].get('train_epochs', 20))
    )
    return data_collector, sentiment_analyzer, market_predictor

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    config = load_config()
    data_collector, sentiment_analyzer, market_predictor = initialize_components(config)
    
    dashboard = TradingDashboard(
        data_collector=data_collector,
        sentiment_analyzer=sentiment_analyzer,
        market_predictor=market_predictor,
        config=config
    )
    dashboard.run()