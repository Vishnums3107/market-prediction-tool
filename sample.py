import yfinance as yf

symbol = "EURUSD=X"  # Replace with the actual symbol you're testing

data = yf.download(symbol, period="5d", interval="1h")
print(data.tail())
