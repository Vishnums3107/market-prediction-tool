import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime
from typing import Dict
from modules.data_collector import DataCollector
from modules.sentiment import SentimentAnalyzer
from modules.predictor import MarketPredictor
import configparser
import talib
class TradingDashboard:
    def __init__(self, data_collector, sentiment_analyzer, market_predictor, config):
        self.data_collector = data_collector
        self.sentiment_analyzer = sentiment_analyzer
        self.market_predictor = market_predictor
        self.config = config
        self._setup_ui()
        
    def _setup_ui(self):
        st.set_page_config(
            page_title="AI Market Predictor",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run(self):
        self.show_sidebar()
        self.show_main_panel()
        
    def show_sidebar(self):
        with st.sidebar:
            st.title("Configuration")
            
            self.symbol = st.text_input("Symbol", "EURUSD=X")
            self.interval = st.selectbox(
                "Interval",
                ["1m", "5m", "15m", "30m", "60m", "1d"],
                index=2
            )
            self.period = st.selectbox(
                "Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                index=2
            )
            
            st.subheader("Model Parameters")
            self.lookback = st.slider(
                "Lookback Period",
                min_value=30,
                max_value=200,
                value=int(self.config['model'].get('lookback_window', 60))
            )
            self.pred_steps = st.slider(
                "Prediction Steps",
                min_value=1,
                max_value=10,
                value=int(self.config['model'].get('prediction_steps', 3))
            )
            
            st.subheader("Trading Strategy")
            self.rsi_overbought = st.slider(
                "RSI Overbought",
                min_value=70,
                max_value=90,
                value=70
            )
            self.rsi_oversold = st.slider(
                "RSI Oversold",
                min_value=10,
                max_value=30,
                value=30
            )
    
    def show_main_panel(self):
        st.title("AI Market Prediction Dashboard")
        
        with st.spinner("Loading data..."):
            df = self.data_collector.fetch_price_data(
                symbol=self.symbol,
                interval=self.interval,
                period=self.period
            )
            
            if df.empty:
                st.error(f"""
                ## Data Loading Failed
                **Symbol:** {self.symbol}  
                **Possible Issues:**
                - Invalid symbol format (try 'AAPL' or 'BTC-USD')
                - Market closed for this asset
                - Yahoo Finance API limitation
                
                **Troubleshooting:**
                1. Check symbol on [Yahoo Finance](https://finance.yahoo.com)
                2. Try popular symbols:
                ```python
                # Stocks
                "TSLA", "MSFT", "GOOG"
                
                # Crypto
                "BTC-USD", "ETH-USD"
                
                # Forex
                "EURUSD=X", "GBPUSD=X"
                ```
                3. Wait 2 minutes and retry
                """)
                df = self.data_collector._generate_dummy_data()
                st.warning("Showing sample data for demonstration")
                
        tab1, tab2, tab3 = st.tabs(["Market View", "Predictions", "Trading Signals"])
        
        with tab1:
            self._show_market_view(df)
            
        with tab2:
            self._show_predictions(df)
            
        with tab3:
            self._show_trading_signals(df)
    
    def _show_market_view(self, df):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color='rgba(100, 100, 255, 0.6)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            title=f"{self.symbol} Price and Volume",
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show technical indicators
        st.subheader("Technical Indicators")
        fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        # Add technical indicators to the figure
        df = self.data_collector._add_technical_indicators(df)
        # RSI
        fig_tech.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name="RSI",
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig_tech.add_hline(
            y=self.rsi_overbought,
            line_dash="dot",
            line_color="red",
            row=1, col=1
        )
        fig_tech.add_hline(
            y=self.rsi_oversold,
            line_dash="dot",
            line_color="green",
            row=1, col=1
        )
        
        # MACD
        fig_tech.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Bollinger Bands
        fig_tech.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name="Price",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        fig_tech.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_upper'],
                name="Upper Band",
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        fig_tech.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_lower'],
                name="Lower Band",
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        
        fig_tech.update_layout(height=900)
        st.plotly_chart(fig_tech, use_container_width=True)
    
    def _show_predictions(self, df):
        with st.spinner("Training model..."):
            X_train, y_train, X_test, y_test = self.market_predictor.prepare_data(df)
            model, history = self.market_predictor.train_model(
                X_train, y_train, X_test, y_test,
                model_name=f"{self.symbol}_{self.interval}"
            )
            
            # Plot training history
            fig_history = go.Figure()
            fig_history.add_trace(
                go.Scatter(
                    y=history.history['loss'],
                    name="Training Loss",
                    mode='lines'
                )
            )
            fig_history.add_trace(
                go.Scatter(
                    y=history.history['val_loss'],
                    name="Validation Loss",
                    mode='lines'
                )
            )
            fig_history.update_layout(
                title="Model Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss"
            )
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Make predictions
            last_seq = X_test[-1:]
            predictions = self.market_predictor.predict(
                model, last_seq, steps=self.pred_steps
            )
            
            # Create future dates
            last_date = df.index[-1]
            freq = f"{int(self.interval[:-1])}min" if self.interval.endswith('m') else 'D'
            future_dates = pd.date_range(
                start=last_date,
                periods=self.pred_steps + 1,
                freq=freq
            )[1:]
            
            # Plot predictions
            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=df.index[-100:],
                    y=df['Close'][-100:],
                    name="Historical Price",
                    mode='lines'
                )
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=predictions,
                    name="Predicted Price",
                    mode='lines+markers',
                    line=dict(color='green', dash='dot')
                )
            )
            fig_pred.update_layout(
                title=f"Price Predictions for Next {self.pred_steps} Periods",
                height=500
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show prediction table
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions
            })
            st.dataframe(pred_df.style.format({'Predicted Price': "{:.4f}"}))
    
    def _show_trading_signals(self, df):
        df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df['Close'])
        df['bollinger_upper'] = upper
        df['bollinger_lower'] = lower
        df['signal'] = 'Hold'
        df.loc[
            (df['rsi'] < self.rsi_oversold) & 
            (df['Close'] < df['bollinger_lower']), 
            'signal'
        ] = 'Buy'
        df.loc[
            (df['rsi'] > self.rsi_overbought) & 
            (df['Close'] > df['bollinger_upper']), 
            'signal'
        ] = 'Sell'
        
        # Plot signals
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name="Price",
                line=dict(color='blue', width=2)
            )
        )
        
        # Add signals
        buy_signals = df[df['signal'] == 'Buy']
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                name="Buy",
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                )
            )
        )
        
        sell_signals = df[df['signal'] == 'Sell']
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                name="Sell",
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                )
            )
        )
        
        fig.update_layout(
            title="Trading Signals",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent signals
        st.subheader("Recent Trading Signals")
        signal_df = df[df['signal'] != 'Hold'].tail(10)
        st.dataframe(signal_df[['Open', 'High', 'Low', 'Close', 'rsi', 'signal']])