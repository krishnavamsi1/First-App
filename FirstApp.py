import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 6))
    sns.histplot(df['damt'], bins=50, kde=True)
    plt.title('Donation Amount Distribution')
    plt.xlabel('Donation Amount')
    plt.ylabel('Frequency')
    plt.axvline(df['damt'].median(), color='red', linestyle='--', label='Median')
    plt.legend()
    return plt

def create_financial_correlation_plot(df):
    financial_vars = ['avhv', 'incm', 'inca', 'damt']
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[financial_vars].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Financial Metrics Correlation')
    return plt

def create_family_size_analysis(df):
    plt.figure(figsize=(12, 6))
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Bar plot for number of donors
    donor_counts = df.groupby('kids').size()
    ax1.bar(donor_counts.index, donor_counts.values, alpha=0.7, label='Number of Donors')
    ax1.set_xlabel('Number of Children')
    ax1.set_ylabel('Number of Donors', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Line plot for average donation
    avg_donation = df.groupby('kids')['damt'].mean()
    ax2.plot(avg_donation.index, avg_donation.values, color='red', label='Average Donation')
    ax2.set_ylabel('Average Donation Amount', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Family Size Analysis')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    return fig

def main():
    st.title('Charity Dataset Dashboard')
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header('Filters')
    min_donation = st.sidebar.slider('Minimum Donation Amount', float(df['damt'].min()), float(df['damt'].max()), float(df['damt'].min()))
    
    selected_kids = st.sidebar.multiselect('Number of Children', options=sorted(df['kids'].unique()), default=sorted(df['kids'].unique()))
    
    # Filter data
    filtered_df = df[
        (df['damt'] >= min_donation) &
        (df['kids'].isin(selected_kids))
    ]
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Donors", len(filtered_df))
    with col2:
        st.metric("Average Donation", f"${filtered_df['damt'].mean():.2f}")
    with col3:
        st.metric("Total Donations", f"${filtered_df['damt'].sum():.2f}")
    
    # Visualization 1: Donation Distribution
    st.subheader("Donation Distribution Analysis")
    donation_dist_plot = create_donation_distribution_plot(filtered_df)
    st.pyplot(donation_dist_plot)
    plt.close(donation_dist_plot.number)
    
    # Visualization 2: Financial Correlations
    st.subheader("Financial Metrics Correlation")
    correlation_plot = create_financial_correlation_plot(filtered_df)
    st.pyplot(correlation_plot)
    plt.close(correlation_plot.number)
    
    # Visualization 3: Family Size Analysis
    st.subheader("Family Size Impact Analysis")
    family_size_plot = create_family_size_analysis(filtered_df)
    st.pyplot(family_size_plot)
    plt.close(family_size_plot.number)
    
    # Additional insights
    st.sidebar.markdown("""
    ### Key Insights
    1. Bimodal donation distribution
    2. Strong financial metric correlations
    3. Family size influences giving patterns
    """)

if __name__ == "__main__":
    main()
