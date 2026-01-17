import streamlit as st
import pandas as pd
import plotly.express as px

# Configure page layout to wide mode
st.set_page_config(
    page_title="XYZ Logistics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Load Data (Simulated loading of your CSV)
@st.cache_data
def load_data():
    # In real app: df = pd.read_csv("CASE_STUDY_DATA.csv")
    # This structure matches your provided file
    df = pd.read_csv('data.csv',delimiter='|')
    # Convert European decimal format (comma) to standard (dot)
    df['metric_value'] = df['metric_value'].astype(str).str.replace(',', '.').astype(float)
    # Convert Excel serial date to datetime (Excel dates start from 1899-12-30)
    df['date_month'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['date_month'], unit='D')
    return df

df = load_data()

# 2. Sidebar Filters
st.sidebar.title("🌍 XYZ Logistics Command")

date_range = st.sidebar.slider(
    "Select Date Range",
    df['date_month'].min().date(),
    df['date_month'].max().date(),
    (df['date_month'].min().date(), df['date_month'].max().date())
)

# Add Metrics Definitions in an expander
with st.sidebar.expander("📚 Metrics Definitions"):
    st.markdown("""
    **Closing**: % of time that the delivery network is closed (i.e. customers cannot place orders) because there aren't enough couriers to fulfill the demand
    
    **Delivery CPO**: Cost per order in euros associated with the delivery of the order
    
    **OPH**: Average number of orders completed by 1 courier per operating hour (orders per hour)
    
    **Orders**: Number of orders delivered by couriers
    
    **PtoD**: Average time in minutes from customer placing an order to courier arriving at the customer's door (placed to delivered)
    
    **UDO**: % of orders that customers claim to not have been delivered and for which a refund is provided
    """)

# Initial filter by date only (country filter moved to sections)
filtered_df = df[
    (df['date_month'].dt.date >= date_range[0]) & 
    (df['date_month'].dt.date <= date_range[1])
]

# Calculate country clusters based on latest period order volume
latest_date_for_clustering = filtered_df['date_month'].max()
latest_orders = filtered_df[
    (filtered_df['date_month'] == latest_date_for_clustering) & 
    (filtered_df['metric'] == 'Orders')
][['country', 'metric_value']].copy()
latest_orders.columns = ['country', 'orders']

# Define cluster thresholds
def assign_cluster(orders):
    if orders >= 50000:
        return "🔵 Large (50K+)"
    elif orders >= 20000:
        return "🟢 Medium (20K-50K)"
    elif orders >= 5000:
        return "🟡 Small (5K-20K)"
    else:
        return "🟠 Tiny (<5K)"

latest_orders['cluster'] = latest_orders['orders'].apply(assign_cluster)
country_clusters = dict(zip(latest_orders['country'], latest_orders['cluster']))

st.title("Executive Performance Overview")

# 1. METRICS SUMMARY TABLE (moved to top)
st.subheader("📊 Latest Period Metrics by Country")

# Add toggle for comparison type
comparison_type = st.radio(
    "Comparison Period:",
    ["Month-over-Month", "Year-to-Date (vs January)"],
    horizontal=True,
    key="comparison_toggle"
)

# Get the most recent date and comparison date in the filtered data
latest_date = filtered_df['date_month'].max()
all_dates = sorted(filtered_df['date_month'].unique())

# Determine comparison date based on selection
if comparison_type == "Month-over-Month":
    comparison_date = all_dates[-2] if len(all_dates) >= 2 else None
    comparison_label = "MoM"
else:  # Year-to-Date
    # Find January of the same year as latest_date
    january_date = pd.Timestamp(year=latest_date.year, month=1, day=1)
    # Find the closest date to January in the data
    january_candidates = [d for d in all_dates if d.year == latest_date.year and d.month == 1]
    comparison_date = january_candidates[0] if january_candidates else None
    comparison_label = "YTD"

# Filter data for the latest period
latest_period_df = filtered_df[filtered_df['date_month'] == latest_date].copy()

if not latest_period_df.empty:
    # Pivot to get metrics as columns for latest month
    metrics_table = latest_period_df.pivot_table(
        index='country',
        columns='metric',
        values='metric_value',
        aggfunc='mean'
    ).reset_index()
    
    # Add cluster information
    metrics_table['Cluster'] = metrics_table['country'].map(country_clusters)
    
    # Sort by Orders descending
    if 'Orders' in metrics_table.columns:
        metrics_table = metrics_table.sort_values('Orders', ascending=False).reset_index(drop=True)
    
    # Calculate % of total orders
    if 'Orders' in metrics_table.columns:
        total_orders = metrics_table['Orders'].sum()
        metrics_table['Orders_pct_total'] = (metrics_table['Orders'] / total_orders * 100)
    
    # Calculate comparison changes if comparison date exists
    comparison_data = {}
    if comparison_date is not None:
        comparison_period_df = filtered_df[filtered_df['date_month'] == comparison_date].copy()
        comparison_metrics = comparison_period_df.pivot_table(
            index='country',
            columns='metric',
            values='metric_value',
            aggfunc='mean'
        )
        
        # Calculate percentage changes
        for metric in ['Orders', 'Delivery CPO', 'OPH', 'PtoD', 'Closing', 'UDO']:
            if metric in metrics_table.columns and metric in comparison_metrics.columns:
                comparison_data[metric] = []
                for country in metrics_table['country']:
                    if country in comparison_metrics.index:
                        current_val = metrics_table[metrics_table['country'] == country][metric].values[0]
                        comparison_val = comparison_metrics.loc[country, metric]
                        if comparison_val != 0:
                            pct_change = ((current_val - comparison_val) / comparison_val) * 100
                            comparison_data[metric].append(pct_change)
                        else:
                            comparison_data[metric].append(None)
                    else:
                        comparison_data[metric].append(None)
    
    # Define metric definitions for tooltips
    metric_definitions = {
        'Orders': 'Number of orders delivered by couriers',
        'Delivery CPO': 'Cost per order in euros associated with the delivery',
        'OPH': 'Average number of orders completed by 1 courier per operating hour',
        'PtoD': 'Average time in minutes from placing order to delivery',
        'Closing': '% of time that the delivery network is closed because there aren\'t enough couriers',
        'UDO': '% of orders claimed undelivered and refunded'
    }
    
    # Prepare data for Plotly table
    header_values = ['<b>Country</b>', '<b>Cluster</b>']
    header_tooltips = ['Country name', 'Market size cluster']
    cell_values = [metrics_table['country'].tolist(), metrics_table['Cluster'].tolist()]
    fill_colors = [['#e8e8e8'] * len(metrics_table), ['#f5f5f5'] * len(metrics_table)]  # Darker background for country column
    column_widths = [120, 180]  # Country and Cluster columns
    cell_align = ['left', 'left']  # Left align names
    cell_font_sizes = [13, 11]  # Font sizes
    cell_font_colors = ['#000000', '#444444']  # Font colors
    
    metric_order = ['Orders', 'Delivery CPO', 'OPH', 'PtoD', 'Closing', 'UDO']
    
    # Define which metrics are "bad when increased" (reverse coloring)
    bad_when_increased = ['Delivery CPO', 'PtoD', 'Closing', 'UDO']
    
    # Helper function to calculate gradient color
    def get_gradient_color(value, is_bad_when_increased):
        """Returns RGB color with gradient based on value magnitude"""
        if value == 0:
            return 'white'
        
        # Determine if this is good or bad
        is_positive = value > 0
        is_good = (is_positive and not is_bad_when_increased) or (not is_positive and is_bad_when_increased)
        
        # Cap the intensity at +/- 20% for color scaling
        intensity = min(abs(value) / 20.0, 1.0)
        
        if is_good:
            # Green gradient: from white (255,255,255) to light green (212,237,218)
            r = int(255 - (255 - 212) * intensity)
            g = int(255 - (255 - 237) * intensity)
            b = int(255 - (255 - 218) * intensity)
        else:
            # Red gradient: from white (255,255,255) to light red (248,215,218)
            r = int(255 - (255 - 248) * intensity)
            g = int(255 - (255 - 215) * intensity)
            b = int(255 - (255 - 218) * intensity)
        
        return f'rgb({r},{g},{b})'
    
    for metric in metric_order:
        if metric in metrics_table.columns:
            # Add metric column
            header_values.append(f'<b>{metric}</b>')
            header_tooltips.append(metric_definitions.get(metric, ''))
            column_widths.append(100)  # Wider for value columns
            cell_align.append('right')  # Right align numbers
            cell_font_sizes.append(12)  # Regular size for metrics
            cell_font_colors.append('#000000')  # Dark color for metrics
            
            # Format values based on metric type
            formatted_values = []
            for val in metrics_table[metric]:
                if pd.isna(val):
                    formatted_values.append('-')
                elif metric == 'Orders':
                    formatted_values.append(f'{int(val):,}')
                elif metric == 'Delivery CPO':
                    formatted_values.append(f'€{val:.2f}')
                elif metric == 'OPH':
                    formatted_values.append(f'{val:.2f}')
                elif metric == 'PtoD':
                    formatted_values.append(f'{int(val)}m')
                elif metric in ['Closing', 'UDO']:
                    formatted_values.append(f'{val * 100:.2f}%')
            
            cell_values.append(formatted_values)
            fill_colors.append(['white'] * len(metrics_table))
            
            # Add % of total after Orders
            if metric == 'Orders' and 'Orders_pct_total' in metrics_table.columns:
                header_values.append('<b>% of Total</b>')
                header_tooltips.append('Country\'s share of total orders in the period')
                column_widths.append(80)
                cell_align.append('right')
                cell_font_sizes.append(10)  # Smaller for secondary metric
                cell_font_colors.append('#666666')  # Lighter gray for secondary
                
                pct_formatted = [f'{val:.1f}%' for val in metrics_table['Orders_pct_total']]
                cell_values.append(pct_formatted)
                fill_colors.append(['white'] * len(metrics_table))
            
            # Add comparison column if data exists
            if metric in comparison_data:
                header_values.append(f'<b>Δ{comparison_label}%</b>')
                if comparison_type == "Month-over-Month":
                    header_tooltips.append('Month-over-month change')
                else:
                    header_tooltips.append('Change vs January (Year-to-Date)')
                column_widths.append(75)  # Narrower for diff columns
                cell_align.append('right')
                cell_font_sizes.append(10)  # Smaller for comparison
                cell_font_colors.append('#666666')  # Lighter gray for comparison
                
                comparison_formatted = []
                comparison_colors = []
                is_bad_metric = metric in bad_when_increased
                
                for comp_val in comparison_data[metric]:
                    if pd.isna(comp_val) or comp_val is None:
                        comparison_formatted.append('-')
                        comparison_colors.append('white')
                    else:
                        comparison_formatted.append(f'{comp_val:+.1f}%')
                        comparison_colors.append(get_gradient_color(comp_val, is_bad_metric))
                
                cell_values.append(comparison_formatted)
                fill_colors.append(comparison_colors)
    
    # Create Plotly table
    import plotly.graph_objects as go
    
    # Create custom line colors for each column to remove left border from MoM columns
    # We'll make vertical lines very light/invisible to create visual grouping
    fig_table = go.Figure(data=[go.Table(
        columnwidth=column_widths,
        header=dict(
            values=header_values,
            fill_color='#f0f2f6',
            align=cell_align,
            font=dict(size=12, color='black'),
            height=40,
            line=dict(width=0.5, color='#f0f0f0')  # Very subtle lines
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            align=cell_align,
            font=dict(size=cell_font_sizes, color=cell_font_colors),  # Per-column font styling
            height=30,
            line=dict(width=0.5, color='#f5f5f5')  # Very subtle lines for grouping effect
        )
    )])
    
    fig_table.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=min(600, 40 + len(metrics_table) * 30)
    )
    
    # Add section headers
    st.markdown("**Scale** | **Delivery Ops** | **User Satisfaction**")
    
    caption_text = f"Data for: {latest_date.strftime('%B %Y')}"
    if comparison_date:
        caption_text += f" (vs. {comparison_date.strftime('%B %Y')})"
    
    if comparison_type == "Month-over-Month":
        caption_text += f" - Δ{comparison_label}% = Month-over-month change • Green = improvement, Red = decline"
    else:
        caption_text += f" - Δ{comparison_label}% = Change vs January • Green = improvement, Red = decline"
    
    st.caption(caption_text)
    
    st.plotly_chart(fig_table, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

# 2. BUBBLE CHART - Metric Relationships
st.subheader("🎯 Metric Relationship Explorer")

# Metric selectors in columns
bubble_col1, bubble_col2 = st.columns(2)

with bubble_col1:
    x_metric = st.selectbox(
        "X-Axis Metric",
        ['OPH', 'Delivery CPO', 'PtoD', 'Closing', 'UDO'],
        index=0,
        key="bubble_x_metric"
    )

with bubble_col2:
    y_metric = st.selectbox(
        "Y-Axis Metric",
        ['PtoD', 'Delivery CPO', 'OPH', 'Closing', 'UDO'],
        index=0,
        key="bubble_y_metric"
    )

# Create bubble chart using latest period data
if not latest_period_df.empty:
    # Pivot to get all metrics for bubble chart
    bubble_data = latest_period_df.pivot_table(
        index='country',
        columns='metric',
        values='metric_value',
        aggfunc='mean'
    ).reset_index()
    
    # Add cluster information
    bubble_data['Cluster'] = bubble_data['country'].map(country_clusters)
    
    # Convert percentage metrics for display
    if x_metric in ['Closing', 'UDO'] and x_metric in bubble_data.columns:
        bubble_data[f'{x_metric}_display'] = bubble_data[x_metric] * 100
        x_col = f'{x_metric}_display'
    else:
        x_col = x_metric
    
    if y_metric in ['Closing', 'UDO'] and y_metric in bubble_data.columns:
        bubble_data[f'{y_metric}_display'] = bubble_data[y_metric] * 100
        y_col = f'{y_metric}_display'
    else:
        y_col = y_metric
    
    # Create bubble chart
    import plotly.graph_objects as go
    
    # Define cluster colors matching the emoji indicators
    cluster_colors = {
        "🔵 Large (50K+)": "#4472C4",      # Blue
        "🟢 Medium (20K-50K)": "#70AD47",  # Green
        "🟡 Small (5K-20K)": "#FFC000",    # Yellow/Gold
        "🟠 Tiny (<5K)": "#ED7D31"         # Orange
    }
    
    fig_bubble = go.Figure()
    
    # Add a trace for each cluster
    for cluster in sorted(set(bubble_data['Cluster'].dropna())):
        cluster_data = bubble_data[bubble_data['Cluster'] == cluster]
        
        fig_bubble.add_trace(go.Scatter(
            x=cluster_data[x_col],
            y=cluster_data[y_col],
            mode='markers+text',
            name=cluster,
            marker=dict(
                size=cluster_data['Orders'] / 1000,  # Scale down for visibility
                sizemode='diameter',
                sizemin=5,
                color=cluster_colors.get(cluster, '#999999'),  # Use cluster-specific color
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=cluster_data['country'],
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_metric}: %{{x:.2f}}<br>' +
                         f'{y_metric}: %{{y:.2f}}<br>' +
                         'Orders: %{marker.size:.0f}K<br>' +
                         '<extra></extra>'
        ))
    
    # Format axis labels
    x_label = f'{x_metric} {"(%)" if x_metric in ["Closing", "UDO"] else ""}'
    y_label = f'{y_metric} {"(%)" if y_metric in ["Closing", "UDO"] else ""}'
    
    if x_metric == 'PtoD':
        x_label += ' (minutes)'
    elif x_metric == 'Delivery CPO':
        x_label += ' (€)'
    elif x_metric == 'OPH':
        x_label += ' (orders/hour)'
    
    if y_metric == 'PtoD':
        y_label += ' (minutes)'
    elif y_metric == 'Delivery CPO':
        y_label += ' (€)'
    elif y_metric == 'OPH':
        y_label += ' (orders/hour)'
    
    fig_bubble.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title="Market Size",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption(f"Bubble size represents Orders volume • Data for: {latest_date.strftime('%B %Y')}")
else:
    st.info("No data available for bubble chart.")

# 3. METRIC TRENDLINE ANALYSIS - 3x2 Grid
st.subheader("📈 Metric Trendlines - All Metrics View")

# Filters in columns (no metric selector - show all)
col1, col2 = st.columns([1, 1])

with col1:
    # Cluster filter
    all_clusters = sorted(set(country_clusters.values()))
    selected_clusters = st.multiselect(
        "Filter by Market Size",
        all_clusters,
        default=all_clusters,
        key="cluster_filter"
    )
    
    # Get countries in selected clusters
    countries_in_clusters = [
        country for country, cluster in country_clusters.items() 
        if cluster in selected_clusters
    ]

with col2:
    # Country multiselect - only show countries from selected clusters
    selected_countries_trend = st.multiselect(
        "Select Countries to Compare",
        sorted(countries_in_clusters),
        default=sorted(countries_in_clusters),
        key="trend_countries"
    )

# Create 3x2 grid of charts
if selected_countries_trend:
    import plotly.graph_objects as go
    
    # Define all metrics and their properties
    metrics_config = {
        'Orders': {
            'title': 'Orders',
            'yaxis_title': 'Number of Orders',
            'format': ',.0f',
            'convert_pct': False,
            'good_direction': 'up'
        },
        'Delivery CPO': {
            'title': 'Delivery CPO',
            'yaxis_title': 'Cost Per Order (€)',
            'format': '€.2f',
            'convert_pct': False,
            'good_direction': 'down'
        },
        'OPH': {
            'title': 'OPH',
            'yaxis_title': 'Orders Per Hour',
            'format': '.2f',
            'convert_pct': False,
            'good_direction': 'up'
        },
        'PtoD': {
            'title': 'PtoD',
            'yaxis_title': 'Time (minutes)',
            'format': '.0f',
            'convert_pct': False,
            'good_direction': 'down'
        },
        'Closing': {
            'title': 'Closing',
            'yaxis_title': 'Percentage (%)',
            'format': '.2f',
            'convert_pct': True,
            'good_direction': 'down'
        },
        'UDO': {
            'title': 'UDO',
            'yaxis_title': 'Percentage (%)',
            'format': '.2f',
            'convert_pct': True,
            'good_direction': 'down'
        }
    }
    
    # Create 3 rows with 2 columns each
    metric_list = ['Orders', 'Delivery CPO', 'OPH', 'PtoD', 'Closing', 'UDO']
    
    for row in range(2):
        cols = st.columns(3)
        
        for col_idx in range(3):
            metric_idx = row * 3 + col_idx
            metric = metric_list[metric_idx]
            config = metrics_config[metric]
            
            with cols[col_idx]:
                # Filter data for this metric
                metric_df = filtered_df[
                    (filtered_df['metric'] == metric) & 
                    (filtered_df['country'].isin(selected_countries_trend))
                ].copy()
                
                # Convert to percentage if needed
                if config['convert_pct']:
                    metric_df['metric_value'] = metric_df['metric_value'] * 100
                
                # Create chart
                fig = go.Figure()
                
                for country in selected_countries_trend:
                    country_data = metric_df[metric_df['country'] == country].sort_values('date_month')
                    
                    if not country_data.empty:
                        fig.add_trace(go.Scatter(
                            x=country_data['date_month'],
                            y=country_data['metric_value'],
                            mode='lines+markers',
                            name=country,
                            line=dict(width=2),
                            marker=dict(size=4),
                            showlegend=(metric_idx == 1)  # Only show legend on top-right chart
                        ))
                
                # Update layout for compact view
                fig.update_layout(
                    title=dict(
                        text=f'<b>{config["title"]}</b>',
                        font=dict(size=14)
                    ),
                    xaxis_title=None,
                    yaxis_title=config['yaxis_title'],
                    yaxis_title_font=dict(size=11),
                    yaxis=dict(rangemode='tozero'),  # Fix y-axis to start at 0
                    hovermode='x unified',
                    height=280,
                    margin=dict(l=50, r=10, t=40, b=30),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1.0,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=9)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f'chart_{metric}')
                
                # Add small caption
                direction_text = "↓ Lower is better" if config['good_direction'] == 'down' else "↑ Higher is better"
                st.caption(direction_text, unsafe_allow_html=True)
    
else:
    st.info("Please select at least one country to view trends.")
