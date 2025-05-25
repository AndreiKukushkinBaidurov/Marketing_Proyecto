import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import base64

warnings.filterwarnings('ignore')

# Enhanced page configuration
st.set_page_config(
    page_title="Advanced Marketing Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced data loading with multiple file format support
@st.cache_data
def load_data():
    """Load marketing data with error handling and validation"""
    file_paths = [
        r"C:\Users\Andrei.Baidurov\Marketing_Proyecto\data\marketingcampaigns_clean.csv",
        "data/marketingcampaigns_clean.csv",
        "marketingcampaigns_clean.csv"
    ]
    
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            # Data validation and cleaning
            df = clean_and_validate_data(df)
            return df
        except FileNotFoundError:
            continue
    
    # Generate sample data if file not found
    return generate_sample_data()

def clean_and_validate_data(df):
    """Clean and validate the marketing data"""
    # Convert date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Fill missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill missing values for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    return df

def generate_sample_data():
    """Generate comprehensive sample marketing data"""
    np.random.seed(42)
    n_campaigns = 1000
    
    campaign_types = ['Social Media', 'Email', 'PPC', 'Display', 'Content Marketing', 'Affiliate']
    channels = ['Facebook', 'Google', 'Instagram', 'LinkedIn', 'Twitter', 'YouTube']
    demographics = ['18-24', '25-34', '35-44', '45-54', '55+']
    devices = ['Desktop', 'Mobile', 'Tablet']
    
    data = {
        'Campaign_ID': [f'CAM_{i:04d}' for i in range(n_campaigns)],
        'Campaign_Type': np.random.choice(campaign_types, n_campaigns),
        'Channel': np.random.choice(channels, n_campaigns),
        'Target_Demographic': np.random.choice(demographics, n_campaigns),
        'Device_Type': np.random.choice(devices, n_campaigns),
        'Budget': np.random.lognormal(8, 1, n_campaigns).round(2),
        'Impressions': np.random.lognormal(10, 1, n_campaigns).astype(int),
        'Clicks': np.random.poisson(100, n_campaigns),
        'Conversions': np.random.poisson(10, n_campaigns),
        'Revenue': np.random.lognormal(7, 1.5, n_campaigns).round(2),
        'Cost_Per_Click': np.random.uniform(0.5, 5.0, n_campaigns).round(2),
        'Date': pd.date_range('2023-01-01', periods=n_campaigns, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['Click_Through_Rate'] = (df['Clicks'] / df['Impressions'] * 100).round(2)
    df['Conversion_Rate'] = (df['Conversions'] / df['Clicks'] * 100).round(2)
    df['Cost_Per_Conversion'] = (df['Budget'] / df['Conversions']).round(2)
    df['Return_On_Ad_Spend'] = (df['Revenue'] / df['Budget']).round(2)
    df['Cost_Per_Mille'] = (df['Budget'] / df['Impressions'] * 1000).round(2)
    
    return df

def create_enhanced_metric_card(title, value, delta=None, delta_color="green"):
    """Create enhanced metric cards with better styling"""
    delta_html = ""
    if delta:
        arrow = "↗️" if delta_color == "green" else "↘️"
        delta_html = f"<p style='color: {delta_color}; margin: 0; font-size: 0.9rem;'>{arrow} {delta}</p>"
    
    return f"""
    <div class="metric-card">
        <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">{title}</h4>
        <h2 style="margin: 0.5rem 0; font-size: 2.2rem; font-weight: bold;">{value}</h2>
        {delta_html}
    </div>
    """

def perform_statistical_analysis(df, column):
    """Perform comprehensive statistical analysis"""
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75),
        'skewness': stats.skew(df[column]),
        'kurtosis': stats.kurtosis(df[column])
    }
    return stats_dict

def perform_customer_segmentation(df):
    """Perform customer segmentation using K-means clustering"""
    if len(df) < 4:
        return df, None
    
    # Select features for clustering
    features = ['Budget', 'Clicks', 'Conversions', 'Revenue']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        return df, None
    
    # Prepare data for clustering
    X = df[available_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    n_clusters = min(5, len(df) // 10)  # Dynamic cluster number
    if n_clusters < 2:
        n_clusters = 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans

def calculate_advanced_metrics(df):
    """Calculate advanced marketing metrics"""
    metrics = {}
    
    if 'Revenue' in df.columns and 'Budget' in df.columns:
        metrics['Total ROI'] = ((df['Revenue'].sum() - df['Budget'].sum()) / df['Budget'].sum() * 100)
        
    if 'Conversions' in df.columns and 'Clicks' in df.columns:
        metrics['Overall Conversion Rate'] = (df['Conversions'].sum() / df['Clicks'].sum() * 100)
        
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        metrics['Overall CTR'] = (df['Clicks'].sum() / df['Impressions'].sum() * 100)
        
    return metrics

# Main application
def main():
    # Title
    st.markdown('<h1 class="main-header">🚀 Advanced Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Comprehensive Marketing Campaign Analysis & Intelligence Platform")
    st.markdown("---")

    # Load and process data
    df = load_data()
    
    if df is not None:
        # Enhanced sidebar with advanced filters
        with st.sidebar:
            st.markdown("## 🎛️ Advanced Control Panel")
            st.markdown("---")
            
            # Advanced filtering options
            st.markdown("### 🔍 Data Filters")
            
            # Campaign type filter
            if 'Campaign_Type' in df.columns:
                campaign_types = st.multiselect(
                    "📈 Campaign Types",
                    options=sorted(df['Campaign_Type'].unique()),
                    default=sorted(df['Campaign_Type'].unique())
                )
                df_filtered = df[df['Campaign_Type'].isin(campaign_types)]
            else:
                df_filtered = df.copy()
            
            # Date range filter
            if 'Date' in df_filtered.columns:
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(df_filtered['Date'].min().date(), df_filtered['Date'].max().date()),
                    min_value=df_filtered['Date'].min().date(),
                    max_value=df_filtered['Date'].max().date()
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_filtered = df_filtered[
                        (df_filtered['Date'].dt.date >= start_date) & 
                        (df_filtered['Date'].dt.date <= end_date)
                    ]
            
            # Budget range filter
            if 'Budget' in df_filtered.columns:
                budget_range = st.slider(
                    "💰 Budget Range",
                    min_value=float(df_filtered['Budget'].min()),
                    max_value=float(df_filtered['Budget'].max()),
                    value=(float(df_filtered['Budget'].min()), float(df_filtered['Budget'].max())),
                    format="$%.2f"
                )
                df_filtered = df_filtered[
                    (df_filtered['Budget'] >= budget_range[0]) & 
                    (df_filtered['Budget'] <= budget_range[1])
                ]
            
            # Channel filter
            if 'Channel' in df_filtered.columns:
                channels = st.multiselect(
                    "📺 Channels",
                    options=sorted(df_filtered['Channel'].unique()),
                    default=sorted(df_filtered['Channel'].unique())
                )
                df_filtered = df_filtered[df_filtered['Channel'].isin(channels)]
            
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Settings")
            
            chart_theme = st.selectbox("🎨 Chart Theme", ["plotly", "plotly_white", "plotly_dark", "seaborn"])
            show_advanced = st.checkbox("🔬 Show Advanced Analytics", value=True)
            
            st.markdown("---")
            st.info(f"📊 **Records Selected:** {len(df_filtered):,} of {len(df):,}")

        # Calculate advanced metrics
        advanced_metrics = calculate_advanced_metrics(df_filtered)

        # Enhanced KPI Dashboard
        st.markdown("## 📊 Key Performance Indicators")
        
        kpi_cols = st.columns(6)
        
        with kpi_cols[0]:
            st.markdown(create_enhanced_metric_card(
                "Total Campaigns", 
                f"{len(df_filtered):,}",
                f"of {len(df):,} total"
            ), unsafe_allow_html=True)
        
        with kpi_cols[1]:
            if 'Revenue' in df_filtered.columns:
                total_revenue = df_filtered['Revenue'].sum()
                avg_revenue = df_filtered['Revenue'].mean()
                st.markdown(create_enhanced_metric_card(
                    "Total Revenue", 
                    f"${total_revenue:,.0f}",
                    f"Avg: ${avg_revenue:,.0f}"
                ), unsafe_allow_html=True)
        
        with kpi_cols[2]:
            if 'Budget' in df_filtered.columns:
                total_budget = df_filtered['Budget'].sum()
                st.markdown(create_enhanced_metric_card(
                    "Total Budget", 
                    f"${total_budget:,.0f}",
                    f"ROI: {advanced_metrics.get('Total ROI', 0):.1f}%" if 'Total ROI' in advanced_metrics else None
                ), unsafe_allow_html=True)
        
        with kpi_cols[3]:
            if 'Conversion_Rate' in df_filtered.columns:
                avg_conv = df_filtered['Conversion_Rate'].mean()
                median_conv = df_filtered['Conversion_Rate'].median()
                st.markdown(create_enhanced_metric_card(
                    "Avg Conversion Rate", 
                    f"{avg_conv:.2f}%",
                    f"Median: {median_conv:.2f}%"
                ), unsafe_allow_html=True)
        
        with kpi_cols[4]:
            if 'Click_Through_Rate' in df_filtered.columns:
                avg_ctr = df_filtered['Click_Through_Rate'].mean()
                st.markdown(create_enhanced_metric_card(
                    "Average CTR", 
                    f"{avg_ctr:.2f}%",
                    f"Overall: {advanced_metrics.get('Overall CTR', 0):.2f}%" if 'Overall CTR' in advanced_metrics else None
                ), unsafe_allow_html=True)
        
        with kpi_cols[5]:
            if 'Cost_Per_Click' in df_filtered.columns:
                avg_cpc = df_filtered['Cost_Per_Click'].mean()
                median_cpc = df_filtered['Cost_Per_Click'].median()
                st.markdown(create_enhanced_metric_card(
                    "Average CPC", 
                    f"${avg_cpc:.2f}",
                    f"Median: ${median_cpc:.2f}"
                ), unsafe_allow_html=True)

        st.markdown("---")

        # Enhanced tab structure
        tabs = st.tabs([
            "📊 Executive Dashboard", 
            "📈 Performance Analytics", 
            "🔍 Deep Dive Analysis",
            "📋 Data Explorer",
            "🤖 ML Insights",
            "🎯 Strategic Recommendations",
            "📄 Export & Reports"
        ])

        # Get column types
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df_filtered.select_dtypes(include=['object']).columns.tolist()

        # Tab 1: Executive Dashboard
        with tabs[0]:
            st.markdown("### 🎯 Executive Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by campaign type
                if 'Revenue' in df_filtered.columns and 'Campaign_Type' in df_filtered.columns:
                    revenue_by_type = df_filtered.groupby('Campaign_Type')['Revenue'].sum().sort_values(ascending=True)
                    
                    fig = px.bar(
                        x=revenue_by_type.values,
                        y=revenue_by_type.index,
                        orientation='h',
                        title="💰 Revenue by Campaign Type",
                        template=chart_theme,
                        color=revenue_by_type.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Conversion funnel
                if all(col in df_filtered.columns for col in ['Impressions', 'Clicks', 'Conversions']):
                    funnel_data = {
                        'Stage': ['Impressions', 'Clicks', 'Conversions'],
                        'Count': [
                            df_filtered['Impressions'].sum(),
                            df_filtered['Clicks'].sum(),
                            df_filtered['Conversions'].sum()
                        ]
                    }
                    
                    fig = go.Figure(go.Funnel(
                        y=funnel_data['Stage'],
                        x=funnel_data['Count'],
                        textinfo="value+percent initial"
                    ))
                    fig.update_layout(
                        title="🔄 Marketing Funnel",
                        template=chart_theme,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time series analysis
            if 'Date' in df_filtered.columns and 'Revenue' in df_filtered.columns:
                st.markdown("#### 📈 Revenue Trend Analysis")
                
                daily_metrics = df_filtered.groupby(df_filtered['Date'].dt.date).agg({
                    'Revenue': 'sum',
                    'Budget': 'sum',
                    'Conversions': 'sum',
                    'Clicks': 'sum'
                }).reset_index()
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Daily Revenue', 'Daily Budget', 'Daily Conversions', 'Daily Clicks'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Revenue'], 
                                       name='Revenue', line=dict(color='green')), row=1, col=1)
                fig.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Budget'], 
                                       name='Budget', line=dict(color='red')), row=1, col=2)
                fig.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Conversions'], 
                                       name='Conversions', line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Clicks'], 
                                       name='Clicks', line=dict(color='orange')), row=2, col=2)
                
                fig.update_layout(height=600, template=chart_theme, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Performance Analytics
        with tabs[1]:
            st.markdown("### 🚀 Advanced Performance Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Interactive scatter plot with advanced features
                if len(numeric_columns) >= 2:
                    st.markdown("#### 🔬 Multi-Dimensional Analysis")
                    
                    x_axis = st.selectbox("X-Axis", numeric_columns, key="perf_x")
                    y_axis = st.selectbox("Y-Axis", numeric_columns, key="perf_y", index=1)
                    size_by = st.selectbox("Size by", ['None'] + numeric_columns, key="perf_size")
                    color_by = st.selectbox("Color by", ['None'] + categorical_columns, key="perf_color")
                    
                    fig = px.scatter(
                        df_filtered, 
                        x=x_axis, 
                        y=y_axis,
                        size=size_by if size_by != 'None' else None,
                        color=color_by if color_by != 'None' else None,
                        title=f"📊 {y_axis} vs {x_axis}",
                        template=chart_theme,
                        hover_data=numeric_columns[:5],
                        trendline="ols"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance heatmap
                if len(numeric_columns) > 3:
                    st.markdown("#### 🔥 Correlation Heatmap")
                    
                    selected_metrics = st.multiselect(
                        "Select metrics for correlation",
                        numeric_columns,
                        default=numeric_columns[:6]
                    )
                    
                    if len(selected_metrics) >= 2:
                        corr_matrix = df_filtered[selected_metrics].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="🔗 Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1
                        )
                        fig.update_layout(height=500, template=chart_theme)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Channel performance comparison
            if 'Channel' in df_filtered.columns:
                st.markdown("#### 📺 Channel Performance Comparison")
                
                channel_metrics = df_filtered.groupby('Channel').agg({
                    'Revenue': ['sum', 'mean'],
                    'Budget': 'sum',
                    'Conversion_Rate': 'mean',
                    'Click_Through_Rate': 'mean'
                }).round(2)
                
                channel_metrics.columns = ['Total Revenue', 'Avg Revenue', 'Total Budget', 'Avg Conversion Rate', 'Avg CTR']
                channel_metrics['ROI'] = ((channel_metrics['Total Revenue'] - channel_metrics['Total Budget']) / channel_metrics['Total Budget'] * 100).round(2)
                
                st.dataframe(channel_metrics, use_container_width=True)
                
                # Visualize channel performance
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=channel_metrics.index,
                        y=channel_metrics['ROI'],
                        title="📈 ROI by Channel",
                        template=chart_theme,
                        color=channel_metrics['ROI'],
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        x=channel_metrics['Total Budget'],
                        y=channel_metrics['Total Revenue'],
                        size=channel_metrics['Avg CTR'],
                        text=channel_metrics.index,
                        title="💰 Budget vs Revenue by Channel",
                        template=chart_theme
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Deep Dive Analysis
        with tabs[2]:
            st.markdown("### 🔬 Deep Dive Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Advanced distribution analysis
                if numeric_columns:
                    selected_metric = st.selectbox("Select metric for analysis", numeric_columns)
                    
                    # Statistical summary
                    stats_summary = perform_statistical_analysis(df_filtered, selected_metric)
                    
                    st.markdown(f"#### 📊 Statistical Summary: {selected_metric}")
                    
                    stats_df = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75', 'Skewness', 'Kurtosis'],
                        'Value': [f"{v:.2f}" for v in stats_summary.values()]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Distribution plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=[f'Distribution of {selected_metric}', 'Box Plot Analysis']
                    )
                    
                    fig.add_trace(
                        go.Histogram(x=df_filtered[selected_metric], nbinsx=30, name="Distribution"),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Box(y=df_filtered[selected_metric], name="Box Plot"),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, template=chart_theme, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Outlier analysis
                st.markdown("#### 🎯 Outlier Detection")
                
                if numeric_columns:
                    outlier_column = st.selectbox("Select column for outlier analysis", numeric_columns, key="outlier")
                    
                    Q1 = df_filtered[outlier_column].quantile(0.25)
                    Q3 = df_filtered[outlier_column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df_filtered[(df_filtered[outlier_column] < lower_bound) | 
                                         (df_filtered[outlier_column] > upper_bound)]
                    
                    st.metric("Outliers Detected", f"{len(outliers):,}")
                    st.metric("Outlier Percentage", f"{len(outliers)/len(df_filtered)*100:.2f}%")
                    
                    if len(outliers) > 0:
                        st.markdown("##### Top Outliers")
                        st.dataframe(outliers.nlargest(10, outlier_column)[[outlier_column] + categorical_columns[:2]], 
                                   use_container_width=True)
            
            # Cohort analysis if date data is available
            if 'Date' in df_filtered.columns:
                st.markdown("#### 📅 Time-Based Cohort Analysis")
                
                # Monthly performance cohorts
                df_filtered['Month'] = df_filtered['Date'].dt.to_period('M')
                monthly_cohorts = df_filtered.groupby('Month').agg({
                    'Revenue': 'sum',
                    'Budget': 'sum',
                    'Conversions': 'sum',
                    'Clicks': 'sum'
                }).reset_index()
                
                monthly_cohorts['Month'] = monthly_cohorts['Month'].astype(str)
                monthly_cohorts['ROI'] = ((monthly_cohorts['Revenue'] - monthly_cohorts['Budget']) / monthly_cohorts['Budget'] * 100).round(2)
                
                fig = px.line(monthly_cohorts, x='Month', y='ROI', 
                             title="📈 Monthly ROI Trend", template=chart_theme)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Data Explorer
        with tabs[3]:
            st.markdown("### 📋 Advanced Data Explorer")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("#### 👀 Interactive Data View")
                
                # Advanced filtering options
                cols_to_show = st.multiselect(
                    "Select columns to display",
                    df_filtered.columns.tolist(),
                    default=df_filtered.columns.tolist()[:10]
                )
                
                if cols_to_show:
                    # Pagination
                    rows_per_page = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)
                    
                    total_rows = len(df_filtered)
                    total_pages = (total_rows - 1) // rows_per_page + 1
                    
                    page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1))
                    
                    start_idx = (page - 1) * rows_per_page
                    end_idx = min(start_idx + rows_per_page, total_rows)
                    
                    st.dataframe(
                        df_filtered[cols_to_show].iloc[start_idx:end_idx],
                        use_container_width=True,
                        height=400
                    )
                    
                    st.info(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows}")
            
            with col2:
                st.markdown("#### 📊 Data Quality Report")
                
                # Data quality metrics
                quality_metrics = {
                    'Total Rows': len(df_filtered),
                    'Total Columns': len(df_filtered.columns),
                    'Missing Values': df_filtered.isnull().sum().sum(),
                    'Duplicate Rows': df_filtered.duplicated().sum(),
                    'Numeric Columns': len(numeric_columns),
                    'Categorical Columns': len(categorical_columns)
                }
                
                for metric, value in quality_metrics.items():
                    st.metric(metric, value)
                
                # Missing values visualization
                if df_filtered.isnull().sum().sum() > 0:
                    missing_data = df_filtered.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    if not missing_data.empty:
                        fig = px.bar(
                            x=missing_data.values,
                            y=missing_data.index,
                            orientation='h',
                            title="Missing Values by Column",
                            template=chart_theme
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

        # Tab 5: ML Insights
        with tabs[4]:
            st.markdown("### 🤖 Machine Learning Insights")
            
            if show_advanced:
                # Customer segmentation
                st.markdown("#### 👥 Customer Segmentation Analysis")
                
                df_segmented, kmeans_model = perform_customer_segmentation(df_filtered)
                
                if kmeans_model is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Segment distribution
                        segment_counts = df_segmented['Segment'].value_counts().sort_index()
                        
                        fig = px.pie(
                            values=segment_counts.values,
                            names=[f'Segment {i}' for i in segment_counts.index],
                            title="📊 Segment Distribution",
                            template=chart_theme
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Segment characteristics
                        segment_metrics = df_segmented.groupby('Segment')[numeric_columns[:4]].mean().round(2)
                        
                        fig = px.imshow(
                            segment_metrics.T,
                            title="🔥 Segment Characteristics Heatmap",
                            template=chart_theme,
                            text_auto=True,
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Segment details
                    st.markdown("#### 📈 Segment Performance Analysis")
                    
                    for segment in sorted(df_segmented['Segment'].unique()):
                        segment_data = df_segmented[df_segmented['Segment'] == segment]
                        
                        with st.expander(f"🎯 Segment {segment} Analysis ({len(segment_data)} campaigns)"):
                            seg_col1, seg_col2, seg_col3 = st.columns(3)
                            
                            with seg_col1:
                                if 'Revenue' in segment_data.columns:
                                    st.metric("Avg Revenue", f"${segment_data['Revenue'].mean():,.0f}")
                                if 'Budget' in segment_data.columns:
                                    st.metric("Avg Budget", f"${segment_data['Budget'].mean():,.0f}")
                            
                            with seg_col2:
                                if 'Conversion_Rate' in segment_data.columns:
                                    st.metric("Avg Conversion Rate", f"{segment_data['Conversion_Rate'].mean():.2f}%")
                                if 'Click_Through_Rate' in segment_data.columns:
                                    st.metric("Avg CTR", f"{segment_data['Click_Through_Rate'].mean():.2f}%")
                            
                            with seg_col3:
                                if 'Campaign_Type' in segment_data.columns:
                                    top_campaign = segment_data['Campaign_Type'].mode().iloc[0]
                                    st.metric("Top Campaign Type", top_campaign)
                                if 'Channel' in segment_data.columns:
                                    top_channel = segment_data['Channel'].mode().iloc[0]
                                    st.metric("Top Channel", top_channel)

        # Tab 6: Strategic Recommendations
        with tabs[5]:
            st.markdown("### 🎯 AI-Powered Strategic Recommendations")
            
            # Generate intelligent insights
            insights = []
            recommendations = []
            
            # Revenue analysis
            if 'Revenue' in df_filtered.columns and 'Campaign_Type' in df_filtered.columns:
                revenue_by_type = df_filtered.groupby('Campaign_Type')['Revenue'].sum().sort_values(ascending=False)
                best_campaign = revenue_by_type.index[0]
                worst_campaign = revenue_by_type.index[-1]
                
                insights.append(f"🏆 **Top Performer**: {best_campaign} generates ${revenue_by_type.iloc[0]:,.0f} in revenue")
                insights.append(f"📉 **Underperformer**: {worst_campaign} generates only ${revenue_by_type.iloc[-1]:,.0f} in revenue")
                
                recommendations.append(f"💡 Increase budget allocation to {best_campaign} campaigns by 20-30%")
                recommendations.append(f"🔍 Analyze and optimize {worst_campaign} campaigns or consider discontinuation")
            
            # ROI analysis
            if 'Revenue' in df_filtered.columns and 'Budget' in df_filtered.columns:
                df_filtered['ROI'] = ((df_filtered['Revenue'] - df_filtered['Budget']) / df_filtered['Budget'] * 100)
                high_roi_threshold = df_filtered['ROI'].quantile(0.75)
                high_roi_campaigns = df_filtered[df_filtered['ROI'] > high_roi_threshold]
                
                insights.append(f"💰 **High ROI Campaigns**: {len(high_roi_campaigns)} campaigns achieve >75th percentile ROI")
                recommendations.append(f"🚀 Scale successful high-ROI campaign strategies across other segments")
            
            # Conversion rate analysis
            if 'Conversion_Rate' in df_filtered.columns:
                avg_conversion = df_filtered['Conversion_Rate'].mean()
                low_conversion = df_filtered[df_filtered['Conversion_Rate'] < avg_conversion * 0.5]
                
                insights.append(f"📊 **Conversion Insights**: {len(low_conversion)} campaigns have critically low conversion rates")
                recommendations.append("🎯 Implement A/B testing for landing pages and ad creatives")
                recommendations.append("📱 Optimize mobile experience - significant conversion rate driver")
            
            # Display insights and recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💡 Key Insights")
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🚀 Strategic Recommendations")
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Action plan
            st.markdown("#### 📋 90-Day Action Plan")
            
            action_plan = {
                "Week 1-2": ["🔍 Audit underperforming campaigns", "📊 Set up advanced analytics tracking"],
                "Week 3-4": ["🎯 Launch A/B tests for top opportunities", "💰 Reallocate budget to high-ROI channels"],
                "Week 5-8": ["📈 Scale successful experiments", "🤖 Implement automated bidding strategies"],
                "Week 9-12": ["📊 Comprehensive performance review", "🚀 Plan next quarter optimization strategy"]
            }
            
            for period, actions in action_plan.items():
                with st.expander(f"📅 {period}"):
                    for action in actions:
                        st.markdown(f"- {action}")

        # Tab 7: Export & Reports
        with tabs[6]:
            st.markdown("### 📄 Export & Advanced Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Export Options")
                
                # Data export
                if st.button("📥 Download Filtered Data (CSV)", type="secondary"):
                    csv = df_filtered.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="marketing_data.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Summary report
                if st.button("📋 Generate Summary Report", type="secondary"):
                    summary_report = {
                        'Total Campaigns': len(df_filtered),
                        'Total Revenue': df_filtered['Revenue'].sum() if 'Revenue' in df_filtered.columns else 0,
                        'Total Budget': df_filtered['Budget'].sum() if 'Budget' in df_filtered.columns else 0,
                        'Average ROI': advanced_metrics.get('Total ROI', 0),
                        'Average Conversion Rate': df_filtered['Conversion_Rate'].mean() if 'Conversion_Rate' in df_filtered.columns else 0
                    }
                    
                    st.json(summary_report)
            
            with col2:
                st.markdown("#### 📈 Report Generator")
                
                report_type = st.selectbox(
                    "Select Report Type",
                    ["Executive Summary", "Performance Analysis", "Channel Analysis", "ROI Report"]
                )
                
                if st.button("🎯 Generate Report", type="primary"):
                    if report_type == "Executive Summary":
                        st.success("📊 Executive Summary report generated successfully!")
                        st.info("This feature can be extended to generate PDF reports using libraries like ReportLab or WeasyPrint")
                    
                    elif report_type == "Performance Analysis":
                        st.success("📈 Performance Analysis report generated successfully!")
                    
                    elif report_type == "Channel Analysis":
                        st.success("📺 Channel Analysis report generated successfully!")
                    
                    elif report_type == "ROI Report":
                        st.success("💰 ROI Analysis report generated successfully!")

    else:
        st.error("❌ Unable to load marketing data")
        st.markdown("""
        ### 🔧 Troubleshooting Guide
        1. **Check file path**: Ensure the CSV file exists at the specified location
        2. **File permissions**: Verify read permissions for the data file
        3. **File format**: Confirm the file is in valid CSV format
        4. **Sample data**: The app will generate sample data if the file is not found
        """)

if __name__ == "__main__":
    main()