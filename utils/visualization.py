import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_forecast(historical_data, results, forecast_days, show_confidence=True):
    """Create interactive forecast plot"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Value'],
        name='Historical Data',
        line=dict(color='blue', width=2),
        mode='lines'
    ))
    
    # Colors for different models
    colors = {'ARIMA': '#FF6B6B', 'Random Forest': '#4ECDC4', 'LSTM': '#9B59B6'}
    
    # Add forecast for each model
    for model_name, result in results.items():
        forecast_dates = pd.date_range(
            historical_data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=result['forecast'],
            name=f'{model_name} Forecast',
            line=dict(color=colors.get(model_name, '#FFA07A'), width=2, dash='dash')
        ))
        
        # Add confidence intervals if enabled
        if show_confidence and len(result['forecast']) > 1:
            error = np.std(result['forecast']) * 1.96
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=result['forecast'] + error,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=result['forecast'] - error,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name=f'{model_name} 95% CI',
                fillcolor=f'rgba({hash(model_name) % 255}, 0, 0, 0.1)'
            ))
    
    fig.update_layout(
        title='📈 Time Series Forecast Results',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_comparison(results):
    """Plot model comparison metrics"""
    
    metrics_data = []
    for model_name, result in results.items():
        actual = result['actual']
        forecast = result['forecast'][:len(actual)]
        
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        
        metrics_data.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2)
        })
    
    df = pd.DataFrame(metrics_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='RMSE',
        x=df['Model'],
        y=df['RMSE'],
        marker_color='#FF6B6B'
    ))
    fig.add_trace(go.Bar(
        name='MAE',
        x=df['Model'],
        y=df['MAE'],
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        title='🏆 Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Error (Lower is Better)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig