import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import os
from typing import Tuple

class MarketPredictor:
    def __init__(self, model_dir: str = 'saved_models', lookback: int = 60, epochs: int = 20):
        self.model_dir = model_dir
        self.lookback = lookback
        self.epochs = epochs
        os.makedirs(model_dir, exist_ok=True)
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        df = self._add_technical_indicators(df)
        
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        feature_cols = [col for col in feature_cols if col in df.columns]
        data = df[feature_cols].values
        
        # Scale features
        feature_scaler = StandardScaler()
        scaled_data = feature_scaler.fit_transform(data)
        self.scalers['features'] = feature_scaler
        
        # Scale close prices separately
        close_scaler = MinMaxScaler()
        close_data = df[['close']].values
        scaled_close = close_scaler.fit_transform(close_data)
        self.scalers['close'] = close_scaler
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, :])
            y.append(scaled_close[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        if 'close' in df.columns:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], _, _ = talib.MACD(df['close'])
            df['bollinger_upper'], _, df['bollinger_lower'] = talib.BBANDS(df['close'])
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df
    
    def build_model(self, input_shape: Tuple) -> Model:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = 'default_model') -> Model:
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, f'{model_name}.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(self.model_dir, f'{model_name}_scalers.pkl'))
        
        return model, history
    
    def predict(self, model: Model, last_sequence: np.ndarray, 
               steps: int = 5) -> np.ndarray:
        predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(steps):
            pred = model.predict(current_seq)[0][0]
            predictions.append(pred)
            
            # Update sequence
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred  # Update close price
            
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scalers['close'].inverse_transform(predictions).flatten()