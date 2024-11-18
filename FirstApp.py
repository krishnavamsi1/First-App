import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure all dependencies are installed
try:
    import plotly
    import seaborn
except ImportError:
    os.system("pip install plotly seaborn")

def load_data():
    # Simulate data based on observed distributions
    np.random.seed(42)
    n_samples = 6000
    
    df = pd.DataFrame({
        'avhv': np.random.normal(200, 50, n_samples),
        'incm': np.random.normal(50, 15, n_samples),
        'inca': np.random.normal(50, 15, n_samples),
        'damt': np.concatenate([
            np.random.exponential(1, 4000),
            np.random.normal(15, 3, 2000)
        ]),
        'kids': np.random.choice([0,1,2,3,4,5], n_samples, p=[0.3,0.1,0.3,0.15,0.1,0.05]),
        'home': np.random.choice([0,1], n_samples, p=[0.1,0.9])
    })
    return df

def create_donation_distribution_plot(df):
    fig = px.histogram(df, x='damt', nbins=50,
                       title='Donation Amount Distribution',labels={'damt': 'Donation Amount', 'count': 'Frequency'})
    fig.add_vline(x=df['damt'].median(), line_dash="dash", line_color="red",annotation_text="Median")
    return fig

def create_financial_correlation_plot(df):
    financial_vars = ['avhv', 'incm', 'inca', 'damt']
    corr = df[financial_vars].corr()
    
    fig = px.imshow(corr,
                    labels=dict(color="Correlation"),
                    x=financial_vars,
                    y=financial_vars,
                    color_continuous_scale='RdBu_r')
    fig.update_layout(title='Financial Metrics Correlation')
    return fig

def create_family_size_analysis(df):
    avg_donation = df.groupby('kids')['damt'].mean().reset_index()
    donor_counts = df.groupby('kids').size().reset_index(name='count')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=donor_counts['kids'], y=donor_counts['count'],
               name="Number of Donors"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=avg_donation['kids'], y=avg_donation['damt'],
                   name="Average Donation", line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(
        title='Family Size Analysis',
        xaxis_title='Number of Children',
        yaxis_title='Number of Donors',
        yaxis2_title='Average Donation Amount'
    )
    return fig

def main():
    st.title('Charity Dataset Dashboard')
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header('Filters')
    min_donation = st.sidebar.slider('Minimum Donation Amount',
                                     float(df['damt'].min()),
                                     float(df['damt'].max()),
        
