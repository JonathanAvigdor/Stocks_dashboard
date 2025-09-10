import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def candlestick(df: pd.DataFrame, title: str):
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig

def line(df: pd.DataFrame, x: str, y: str, title: str):
    return px.line(df, x=x, y=y, title=title)
