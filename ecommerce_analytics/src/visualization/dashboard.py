"""Streamlit dashboard for E-commerce Analytics Platform."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import requests
import json
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import db_session
from src.models.entities import Customer, Order, Product, Review, FeatureStat
from src.models.customer_segmentation import CustomerSegmenter
from src.features.customer_features import CustomerFeatureEngineer

# Set up configuration and logging
logger = get_logger(__name__)
config = get_config()

# API base URL
API_BASE_URL = "http://api:8000"  # When running in Docker
# For local development, use:
# API_BASE_URL = "http://localhost:8000"


# Helper functions for API calls
def api_get(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Make a GET request to the API.
    
    Args:
        endpoint: API endpoint (without base URL)
        params: Optional query parameters
        
    Returns:
        Response data
    """
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}


def api_post(endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    """Make a POST request to the API.
    
    Args:
        endpoint: API endpoint (without base URL)
        data: Optional request body
        params: Optional query parameters
        
    Returns:
        Response data
    """
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        response = requests.post(url, json=data, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}


# Dashboard components
def sidebar():
    """Render the sidebar."""
    st.sidebar.title("E-commerce Analytics")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Customer Segmentation", "Customer Profiles", "Orders Analysis"]
    )
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    today = datetime.now().date()
    start_date = st.sidebar.date_input("Start Date", today - timedelta(days=90))
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
    
    # Data refresh button
    if st.sidebar.button("Refresh Data"):
        st.sidebar.success("Data refreshed!")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard visualizes e-commerce data from Shopify, "
        "providing insights into customer behavior, segmentation, "
        "and order patterns."
    )
    
    return page, start_date, end_date


def overview_page():
    """Render the overview page."""
    st.title("E-commerce Analytics Dashboard")
    st.subheader("Business Overview")
    
    # Load customer stats
    customer_stats = api_get("/customers/stats/summary")
    
    if not customer_stats:
        st.warning("Unable to load customer statistics")
        return
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{customer_stats.get('total_customers', 0):,}")
        st.metric("New Customers (30d)", f"{customer_stats.get('new_customers_30d', 0):,}")
    
    with col2:
        st.metric("Customers with Orders", f"{customer_stats.get('customers_with_orders', 0):,}")
        st.metric("Avg. Orders per Customer", f"{customer_stats.get('average_orders_per_customer', 0):.2f}")
    
    with col3:
        st.metric("Avg. Order Value", f"${customer_stats.get('average_order_value', 0):.2f}")
        conversion_rate = customer_stats.get('customers_with_orders', 0) / max(customer_stats.get('total_customers', 1), 1) * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    st.markdown("---")
    
    # Sample data for charts (in a real scenario, this would come from the API)
    # Monthly sales trend
    st.subheader("Monthly Sales Trend")
    
    # In a real scenario, fetch this data from API
    # For now, generate sample data
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sales = np.random.normal(100000, 20000, 12).astype(int)
    sales_data = pd.DataFrame({"Month": months, "Sales": sales})
    
    # Create chart
    sales_chart = alt.Chart(sales_data).mark_line(point=True).encode(
        x=alt.X("Month:N", sort=months),
        y=alt.Y("Sales:Q"),
        tooltip=["Month", "Sales"]
    ).properties(height=300)
    
    st.altair_chart(sales_chart, use_container_width=True)
    
    # Customer acquisition and churn
    st.subheader("Customer Acquisition & Retention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer acquisition
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        new_customers = np.random.normal(500, 100, 6).astype(int)
        acquisition_data = pd.DataFrame({"Month": months, "New Customers": new_customers})
        
        acq_chart = alt.Chart(acquisition_data).mark_bar().encode(
            x="Month:N",
            y="New Customers:Q",
            color=alt.value("#1f77b4"),
            tooltip=["Month", "New Customers"]
        ).properties(title="New Customer Acquisition")
        
        st.altair_chart(acq_chart, use_container_width=True)
    
    with col2:
        # Customer retention (90-day cohort)
        months = ["Month 1", "Month 2", "Month 3"]
        retention = [100, 65, 42]
        retention_data = pd.DataFrame({"Month": months, "Retention %": retention})
        
        ret_chart = alt.Chart(retention_data).mark_bar().encode(
            x="Month:N",
            y="Retention %:Q",
            color=alt.value("#ff7f0e"),
            tooltip=["Month", "Retention %"]
        ).properties(title="90-Day Cohort Retention")
        
        st.altair_chart(ret_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Product performance
    st.subheader("Top Products by Revenue")
    
    # Sample data for top products
    products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    revenue = np.random.normal(10000, 3000, 5).astype(int)
    products_data = pd.DataFrame({"Product": products, "Revenue": revenue})
    products_data = products_data.sort_values("Revenue", ascending=False)
    
    # Create chart
    product_chart = alt.Chart(products_data).mark_bar().encode(
        x=alt.X("Revenue:Q"),
        y=alt.Y("Product:N", sort="-x"),
        tooltip=["Product", "Revenue"]
    ).properties(height=300)
    
    st.altair_chart(product_chart, use_container_width=True)


def customer_segmentation_page():
    """Render the customer segmentation page."""
    st.title("Customer Segmentation")
    
    # Controls for segmentation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Number of Segments", min_value=2, max_value=8, value=4)
    
    with col2:
        limit = st.slider("Number of Customers", min_value=100, max_value=10000, value=1000, step=100)
    
    with col3:
        st.write("")  # Empty space for alignment
        run_button = st.button("Run Segmentation")
    
    # Run segmentation when button is clicked
    if run_button:
        with st.spinner("Running customer segmentation..."):
            segments_data = api_get(
                "/customers/segmentation/segments",
                params={"n_clusters": n_clusters, "limit": limit}
            )
        
        if not segments_data or "error" in segments_data:
            st.error(segments_data.get("error", "Failed to run segmentation"))
        else:
            st.success(f"Segmentation completed for {segments_data.get('total_customers', 0)} customers")
            
            # Display segment statistics
            segment_stats = segments_data.get("segment_stats", {})
            
            # Convert to DataFrame for easier visualization
            segment_rows = []
            for segment_id, stats in segment_stats.items():
                row = {"Segment": int(segment_id), "Count": stats["count"], "Percentage": stats["percentage"]}
                
                # Add profile features
                for feature, value in stats["profile"].items():
                    row[feature] = value
                
                segment_rows.append(row)
            
            if segment_rows:
                segments_df = pd.DataFrame(segment_rows)
                
                # Visualize segment sizes
                st.subheader("Segment Sizes")
                
                fig = px.pie(
                    segments_df, 
                    values="Count", 
                    names="Segment",
                    title="Customer Distribution by Segment",
                    hole=0.4
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Segment profiles
                st.subheader("Segment Profiles")
                
                # Create radar chart for each segment
                feature_columns = [col for col in segments_df.columns if col not in ['Segment', 'Count', 'Percentage']]
                
                # Normalize the feature values for better radar visualization
                segments_df_radar = segments_df.copy()
                for col in feature_columns:
                    min_val = segments_df[col].min()
                    max_val = segments_df[col].max()
                    if max_val > min_val:
                        segments_df_radar[col] = (segments_df[col] - min_val) / (max_val - min_val)
                
                # Create radar chart
                fig = go.Figure()
                
                for _, row in segments_df_radar.iterrows():
                    segment_id = int(row['Segment'])
                    
                    # Add a trace for each segment
                    fig.add_trace(go.Scatterpolar(
                        r=[row[col] for col in feature_columns],
                        theta=feature_columns,
                        fill='toself',
                        name=f'Segment {segment_id}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Segment Profiles (Normalized Features)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show segment profile table
                st.subheader("Segment Profile Details")
                
                # Format the table for better readability
                display_df = segments_df.copy()
                display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                for col in feature_columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df.set_index('Segment'))
                
                # Segment descriptions
                st.subheader("Segment Interpretations")
                
                # Sample segment descriptions (in a real app, these would be generated dynamically)
                segment_desc = {
                    0: "High Value Customers: These customers have high monetary value and frequent purchases.",
                    1: "Loyal Customers: These customers make frequent purchases but with moderate order values.",
                    2: "New Customers: These customers have made recent first purchases.",
                    3: "At Risk Customers: These customers have high historical value but haven't purchased recently."
                }
                
                for segment_id in range(min(n_clusters, 4)):
                    if segment_id in segment_desc:
                        st.write(f"**Segment {segment_id}**: {segment_desc.get(segment_id, '')}")


def customer_profiles_page():
    """Render the customer profiles page."""
    st.title("Customer Profiles")
    
    # Search for a customer
    st.subheader("Customer Lookup")
    search_term = st.text_input("Search by Customer ID, Email, or Name")
    
    if search_term:
        # In a real scenario, this would search the API
        st.write(f"Searching for: {search_term}")
        st.info("This is a placeholder. In a real application, this would search for customers matching the query.")
    
    # Customer profile viewer
    st.subheader("Sample Customer Profile")
    
    # Sample customer data
    customer_data = {
        "id": "01234567-89ab-cdef-0123-456789abcdef",
        "email": "john.doe@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "created_at": "2022-05-15T10:30:00Z",
        "features": {
            "recency_days": 7.5,
            "frequency": 12,
            "monetary": 1250.0,
            "avg_order_value": 104.17,
            "std_order_value": 25.5,
            "days_between_orders": 30.2,
            "customer_lifetime_value": 1458.33,
            "segment": 0
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Customer Information**")
        st.write(f"**Name:** {customer_data['first_name']} {customer_data['last_name']}")
        st.write(f"**Email:** {customer_data['email']}")
        st.write(f"**Customer Since:** {datetime.fromisoformat(customer_data['created_at'].replace('Z', '+00:00')).strftime('%B %d, %Y')}")
        
        # Customer segment
        segment = int(customer_data['features'].get('segment', 0))
        segment_descriptions = {
            0: "High Value Customer",
            1: "Loyal Customer",
            2: "New Customer",
            3: "At Risk Customer"
        }
        
        segment_desc = segment_descriptions.get(segment, f"Segment {segment}")
        st.write(f"**Segment:** {segment_desc}")
    
    with col2:
        st.write("**Customer Metrics**")
        st.write(f"**Total Spend:** ${customer_data['features']['monetary']:.2f}")
        st.write(f"**Purchase Frequency:** {customer_data['features']['frequency']} orders")
        st.write(f"**Average Order Value:** ${customer_data['features']['avg_order_value']:.2f}")
        st.write(f"**Days Since Last Order:** {customer_data['features']['recency_days']:.1f} days")
        st.write(f"**Customer Lifetime Value:** ${customer_data['features']['customer_lifetime_value']:.2f}")
    
    # RFM quadrant visualization
    st.subheader("Customer RFM Analysis")
    
    # Sample RFM data for visualization
    rfm_data = pd.DataFrame({
        'Recency': np.random.normal(30, 20, 100),
        'Frequency': np.random.normal(5, 3, 100),
        'Monetary': np.random.normal(100, 50, 100),
        'Segment': np.random.randint(0, 4, 100)
    })
    
    # Add the sample customer
    sample_customer = pd.DataFrame({
        'Recency': [customer_data['features']['recency_days']],
        'Frequency': [customer_data['features']['frequency']],
        'Monetary': [customer_data['features']['monetary']],
        'Segment': [segment]
    })
    
    # Create scatter plot
    fig = px.scatter(
        rfm_data,
        x='Recency',
        y='Frequency',
        size='Monetary',
        color='Segment',
        opacity=0.5,
        title='Customer RFM Segmentation',
        labels={
            'Recency': 'Days Since Last Purchase',
            'Frequency': 'Number of Orders',
            'Monetary': 'Total Spend'
        }
    )
    
    # Add the selected customer as a distinct point
    fig.add_trace(
        go.Scatter(
            x=[sample_customer['Recency'][0]],
            y=[sample_customer['Frequency'][0]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Selected Customer',
            hoverinfo='name'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Purchase history
    st.subheader("Purchase History")
    
    # Sample order data
    orders = [
        {"order_number": "1001", "date": "2022-06-12", "total": 89.99, "status": "Completed"},
        {"order_number": "1045", "date": "2022-07-25", "total": 129.95, "status": "Completed"},
        {"order_number": "1067", "date": "2022-08-14", "total": 76.50, "status": "Completed"},
        {"order_number": "1105", "date": "2022-10-05", "total": 152.75, "status": "Completed"},
        {"order_number": "1142", "date": "2022-11-20", "total": 95.25, "status": "Completed"},
        {"order_number": "1196", "date": "2023-01-15", "total": 112.50, "status": "Completed"}
    ]
    
    orders_df = pd.DataFrame(orders)
    orders_df['date'] = pd.to_datetime(orders_df['date'])
    orders_df = orders_df.sort_values('date', ascending=False)
    
    st.dataframe(orders_df)


def orders_analysis_page():
    """Render the orders analysis page."""
    st.title("Orders Analysis")
    
    # Order volume and revenue trends
    st.subheader("Order Volume & Revenue Trends")
    
    # Sample data for visualization
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    orders_data = pd.DataFrame({
        'date': dates,
        'order_count': np.random.normal(50, 10, len(dates)) + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates))),
        'revenue': np.random.normal(5000, 1000, len(dates)) + 1000 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    })
    
    # Resample to monthly for better visualization
    orders_monthly = orders_data.set_index('date').resample('M').sum().reset_index()
    orders_monthly['month'] = orders_monthly['date'].dt.strftime('%b %Y')
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Order Volume", "Revenue"])
    
    with tab1:
        fig = px.line(
            orders_monthly,
            x='month',
            y='order_count',
            title='Monthly Order Volume',
            markers=True
        )
        fig.update_layout(xaxis_title='Month', yaxis_title='Number of Orders')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(
            orders_monthly,
            x='month',
            y='revenue',
            title='Monthly Revenue',
            markers=True
        )
        fig.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Order metrics
    st.subheader("Order Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_orders = int(orders_data['order_count'].sum())
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col2:
        avg_order_value = orders_data['revenue'].sum() / total_orders
        st.metric("Average Order Value", f"${avg_order_value:.2f}")
    
    with col3:
        total_revenue = orders_data['revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    # Order distribution by day of week and hour
    st.subheader("Order Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Orders by day of week
        dow_data = orders_data.copy()
        dow_data['dow'] = dow_data['date'].dt.day_name()
        dow_data = dow_data.groupby('dow')['order_count'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig = px.bar(
            x=dow_data.index,
            y=dow_data.values,
            title='Orders by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Number of Orders'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Orders by hour of day
        # Generate hourly data
        hours = np.arange(24)
        hour_data = pd.DataFrame({
            'hour': hours,
            'order_count': 20 + 30 * np.exp(-((hours - 14) ** 2) / 20)  # Bell curve centered at 2 PM
        })
        
        fig = px.line(
            hour_data,
            x='hour',
            y='order_count',
            title='Orders by Hour of Day',
            labels={'hour': 'Hour of Day', 'order_count': 'Average Orders'}
        )
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    
    # Popular products
    st.subheader("Popular Products")
    
    # Sample product data
    products = ["Product A", "Product B", "Product C", "Product D", "Product E", 
               "Product F", "Product G", "Product H", "Product I", "Product J"]
    quantities = np.random.randint(50, 500, size=len(products))
    products_data = pd.DataFrame({
        'product': products,
        'quantity_sold': quantities,
        'revenue': quantities * np.random.uniform(10, 100, size=len(products))
    }).sort_values('quantity_sold', ascending=False)
    
    tab1, tab2 = st.tabs(["By Quantity", "By Revenue"])
    
    with tab1:
        fig = px.bar(
            products_data.sort_values('quantity_sold', ascending=False).head(10),
            y='product',
            x='quantity_sold',
            orientation='h',
            title='Top Products by Quantity Sold',
            labels={'product': 'Product', 'quantity_sold': 'Quantity Sold'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(
            products_data.sort_values('revenue', ascending=False).head(10),
            y='product',
            x='revenue',
            orientation='h',
            title='Top Products by Revenue',
            labels={'product': 'Product', 'revenue': 'Revenue ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(
        page_title="E-commerce Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Render sidebar and get selected page
    page, start_date, end_date = sidebar()
    
    # Render the selected page
    if page == "Overview":
        overview_page()
    elif page == "Customer Segmentation":
        customer_segmentation_page()
    elif page == "Customer Profiles":
        customer_profiles_page()
    elif page == "Orders Analysis":
        orders_analysis_page()


if __name__ == "__main__":
    main() 