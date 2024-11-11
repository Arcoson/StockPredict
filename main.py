#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

console = Console()

def show_welcome():
    welcome_text = """
# 📈 StockPredict

Welcome to StockPredict! This tool helps you analyze and predict stock market trends using AI.

## Commands:
- `analyze`: Get detailed analysis and prediction for a stock
- `help`: Show this help message
- `exit`: Exit the program

## Features:
- Historical price analysis
- Technical indicators
- Financial metrics
- AI-powered price predictions
- Interactive candlestick charts
"""
    console.print(Markdown(welcome_text))

def get_stock_data(symbol: str, period: str = "2y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return stock, hist
    except Exception as e:
        console.print(f"[red]Error fetching stock data: {str(e)}[/]")
        return None, None

def calculate_technical_indicators(df):
    # Calculate 20-day SMA
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate 50-day SMA
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def create_stock_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Add SMAs
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA20'],
        name='SMA20',
        line=dict(color='#1976D2', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA50'],
        name='SMA50',
        line=dict(color='#FB8C00', width=1)
    ), row=1, col=1)

    # Volume bars
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        yaxis_title='Stock Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800
    )

    return fig

def predict_stock_prices(df, days=30):
    # Prepare data for Prophet
    df_prophet = pd.DataFrame()
    df_prophet['ds'] = df.index
    df_prophet['y'] = df['Close']
    
    # Check if we have enough valid data
    if df_prophet['y'].notna().sum() < 2:
        console.print("[red]Error: Insufficient data for prediction. The stock may be too new or have missing data.[/]")
        return None
    
    # Remove any NaN values
    df_prophet = df_prophet.dropna()
    
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future_dates = model.make_future_dataframe(periods=days)
        forecast = model.predict(future_dates)
        
        return forecast
    except Exception as e:
        console.print(f"[red]Error during prediction: {str(e)}[/]")
        return None

def display_financial_metrics(stock):
    info = stock.info
    
    metrics_table = Table(title="Financial Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "EPS": info.get('trailingEps', 'N/A'),
        "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
        "50 Day Avg": info.get('fiftyDayAverage', 'N/A'),
        "200 Day Avg": info.get('twoHundredDayAverage', 'N/A'),
        "Volume": info.get('volume', 'N/A'),
        "Avg Volume": info.get('averageVolume', 'N/A')
    }
    
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            if metric == "Market Cap":
                value = f"${value:,.0f}"
            elif "Volume" in metric:
                value = f"{value:,.0f}"
            else:
                value = f"{value:.2f}"
        metrics_table.add_row(metric, str(value))
    
    console.print(metrics_table)

def analyze_command():
    symbol = Prompt.ask("Enter stock symbol (e.g., AAPL)").upper()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Fetching stock data...", total=100)
        
        stock, hist = get_stock_data(symbol)
        if stock is None or hist is None:
            return
        
        progress.update(task, advance=30)
        progress.update(task, description="Calculating indicators...")
        
        hist = calculate_technical_indicators(hist)
        progress.update(task, advance=20)
        
        progress.update(task, description="Generating predictions...")
        forecast = predict_stock_prices(hist)
        progress.update(task, advance=25)
        
        progress.update(task, description="Creating visualizations...")
        fig = create_stock_chart(hist, symbol)
        fig.show()
        progress.update(task, advance=25)
    
    console.print(Panel.fit(
        f"[bold green]Analysis Complete for {symbol}[/]",
        border_style="green"
    ))
    
    display_financial_metrics(stock)
    
    # Display prediction summary only if forecast is available
    if forecast is not None:
        last_price = hist['Close'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        price_change = ((predicted_price - last_price) / last_price) * 100
        
        prediction_panel = Panel.fit(
            f"[bold]30-Day Price Prediction[/]\n"
            f"Current Price: [green]${last_price:.2f}[/]\n"
            f"Predicted Price: [cyan]${predicted_price:.2f}[/]\n"
            f"Expected Change: [{'green' if price_change >= 0 else 'red'}]{price_change:+.2f}%[/]",
            title="AI Prediction",
            border_style="cyan"
        )
        console.print(prediction_panel)

def main():
    show_welcome()
    
    while True:
        command = Prompt.ask("\nEnter command", choices=["analyze", "help", "exit"])
        
        if command == "analyze":
            analyze_command()
        elif command == "help":
            show_welcome()
        elif command == "exit":
            console.print("[yellow]Thanks for using StockPredict! Goodbye! 📈[/]")
            break

if __name__ == "__main__":
    main()
