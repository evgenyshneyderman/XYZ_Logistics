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

def assign_cluster(orders):
    """
    Assign market size cluster based on order volume.
    
    Args:
        orders: Number of orders
    
    Returns:
        Cluster label string
    """
    if orders >= 50000:
        return "⭐ Star (50K+)"
    elif orders >= 20000:
        return "🔵 L (20K-50K)"
    elif orders >= 5000:
        return "🟢 M (5K-20K)"
    else:
        return "🟡 S (<5K)"

def calculate_percent_of_total(metrics_table):
    """
    Calculate % of total orders for each country (excluding Total row).
    
    Args:
        metrics_table: DataFrame with 'country' and 'Orders' columns
    
    Returns:
        Series with % of total for each country (None for Total row)
    """
    if 'Orders' not in metrics_table.columns:
        return pd.Series([None] * len(metrics_table))
    
    # Calculate total only from non-Total countries
    total_orders_excluding_total = metrics_table[metrics_table['country'] != 'Total']['Orders'].sum()
    
    if total_orders_excluding_total == 0:
        return pd.Series([None] * len(metrics_table))
    
    # Calculate percentage for each country (Total will get None)
    return metrics_table.apply(
        lambda row: (row['Orders'] / total_orders_excluding_total * 100) if row['country'] != 'Total' else None,
        axis=1
    )

@st.cache_data
def calculate_weighted_growth(df):
    """
    Calculate weighted average growth rate for Orders over the last 12 months.
    Recent months have higher weight to capture trend changes.
    """
    # Filter for Orders metric only
    orders_df = df[df['metric'] == 'Orders'].copy()
    
    # Sort by country and date
    orders_df = orders_df.sort_values(['country', 'date_month'])
    
    # Calculate month-over-month growth rate for each country
    orders_df['mom_growth'] = orders_df.groupby('country')['metric_value'].pct_change() * 100
    
    # For each country and date, calculate weighted average of last 12 months
    growth_records = []
    
    for country in orders_df['country'].unique():
        country_data = orders_df[orders_df['country'] == country].sort_values('date_month')
        
        for idx, row in country_data.iterrows():
            current_date = row['date_month']
            
            # Get the last 12 months of growth data (including current month)
            lookback_data = country_data[
                (country_data['date_month'] <= current_date) & 
                (country_data['date_month'] > current_date - pd.DateOffset(months=12))
            ].copy()
            
            # Skip if we don't have at least 2 data points (need for growth calculation)
            if len(lookback_data) < 2:
                continue
            
            # Remove NaN growth values (first month has no previous month)
            lookback_data = lookback_data.dropna(subset=['mom_growth'])
            
            if len(lookback_data) == 0:
                continue
            
            # Create linear weights: older months get lower weight, recent months get higher weight
            # Weights: 1, 2, 3, ..., n (where n is the most recent month)
            n_months = len(lookback_data)
            weights = list(range(1, n_months + 1))
            
            # Calculate weighted average
            weighted_growth = (lookback_data['mom_growth'].values * weights).sum() / sum(weights)
            
            growth_records.append({
                'country': country,
                'date_month': current_date,
                'metric': 'Growth Index',
                'metric_value': weighted_growth / 100  # Convert to decimal to match other metrics
            })
    
    # Create DataFrame with growth rate data
    growth_df = pd.DataFrame(growth_records)
    
    # Append to original dataframe
    df_with_growth = pd.concat([df, growth_df], ignore_index=True)
    
    return df_with_growth

@st.cache_data
def calculate_totals(df):
    """
    Calculate Total (aggregate) metrics across all countries for each date.
    
    Formulas used:
    - Orders: Simple sum across all countries
    - Growth Index: Calculated directly from Total orders (NOT aggregated from countries)
    - Delivery CPO: Weighted average by orders (total delivery cost / total orders)
    - PtoD: Weighted average by orders (represents average customer experience)
    - UDO: Weighted average by orders (total undelivered orders / total orders)
    - Closing & OPH: NOT calculated (require time-series and courier-level data not available)
    """
    total_records = []
    
    # Get all unique dates
    dates = sorted(df['date_month'].unique())
    
    # First pass: Calculate total orders for each date
    total_orders_by_date = {}
    for date in dates:
        date_df = df[df['date_month'] == date]
        orders_data = date_df[date_df['metric'] == 'Orders']
        if not orders_data.empty:
            total_orders = orders_data['metric_value'].sum()
            total_orders_by_date[date] = total_orders
            total_records.append({
                'country': 'Total',
                'date_month': date,
                'metric': 'Orders',
                'metric_value': total_orders
            })
    
    # Second pass: Calculate Growth Index for Total using Total orders history
    total_orders_df = pd.DataFrame([
        {'date_month': date, 'metric_value': orders}
        for date, orders in sorted(total_orders_by_date.items())
    ])
    
    # Calculate MoM growth for Total
    total_orders_df['mom_growth'] = total_orders_df['metric_value'].pct_change() * 100
    
    # Calculate weighted growth index for Total
    for idx, row in total_orders_df.iterrows():
        current_date = row['date_month']
        
        # Get the last 12 months of growth data
        lookback_data = total_orders_df[
            (total_orders_df['date_month'] <= current_date) & 
            (total_orders_df['date_month'] > current_date - pd.DateOffset(months=12))
        ].copy()
        
        if len(lookback_data) < 2:
            continue
        
        lookback_data = lookback_data.dropna(subset=['mom_growth'])
        
        if len(lookback_data) == 0:
            continue
        
        # Linear weights: 1, 2, 3, ..., n
        n_months = len(lookback_data)
        weights = list(range(1, n_months + 1))
        
        weighted_growth = (lookback_data['mom_growth'].values * weights).sum() / sum(weights)
        
        total_records.append({
            'country': 'Total',
            'date_month': current_date,
            'metric': 'Growth Index',
            'metric_value': weighted_growth / 100  # Convert to decimal
        })
    
    # Third pass: Calculate other weighted metrics
    for date in dates:
        date_df = df[df['date_month'] == date]
        
        if date not in total_orders_by_date:
            continue
        
        total_orders = total_orders_by_date[date]
        orders_data = date_df[date_df['metric'] == 'Orders']
        
        if total_orders == 0:
            continue
        
        # Calculate Delivery CPO (weighted average by orders)
        cpo_data = date_df[date_df['metric'] == 'Delivery CPO']
        if not cpo_data.empty:
            cpo_with_orders = cpo_data.merge(
                orders_data[['country', 'metric_value']], 
                on='country', 
                suffixes=('_cpo', '_orders')
            )
            weighted_cpo = (
                cpo_with_orders['metric_value_cpo'] * cpo_with_orders['metric_value_orders']
            ).sum() / total_orders
            
            total_records.append({
                'country': 'Total',
                'date_month': date,
                'metric': 'Delivery CPO',
                'metric_value': weighted_cpo
            })
        
        # Calculate PtoD (weighted average by orders)
        ptod_data = date_df[date_df['metric'] == 'PtoD']
        if not ptod_data.empty:
            ptod_with_orders = ptod_data.merge(
                orders_data[['country', 'metric_value']], 
                on='country', 
                suffixes=('_ptod', '_orders')
            )
            weighted_ptod = (
                ptod_with_orders['metric_value_ptod'] * ptod_with_orders['metric_value_orders']
            ).sum() / total_orders
            
            total_records.append({
                'country': 'Total',
                'date_month': date,
                'metric': 'PtoD',
                'metric_value': weighted_ptod
            })
        
        # Calculate UDO (weighted average by orders)
        udo_data = date_df[date_df['metric'] == 'UDO']
        if not udo_data.empty:
            udo_with_orders = udo_data.merge(
                orders_data[['country', 'metric_value']], 
                on='country', 
                suffixes=('_udo', '_orders')
            )
            weighted_udo = (
                udo_with_orders['metric_value_udo'] * udo_with_orders['metric_value_orders']
            ).sum() / total_orders
            
            total_records.append({
                'country': 'Total',
                'date_month': date,
                'metric': 'UDO',
                'metric_value': weighted_udo
            })
    
    # Create DataFrame with total data
    total_df = pd.DataFrame(total_records)
    
    # Append to original dataframe
    df_with_totals = pd.concat([df, total_df], ignore_index=True)
    
    return df_with_totals

df = load_data()
df = calculate_weighted_growth(df)
df = calculate_totals(df)

# 2. Sidebar Filters
st.sidebar.title("🌍 XYZ Logistics Command")

date_range = st.sidebar.slider(
    "Select Date Range",
    df['date_month'].min().date(),
    df['date_month'].max().date(),
    (df['date_month'].min().date(), df['date_month'].max().date())
)

# Dialog for detailed metrics definitions
@st.dialog("📚 Metrics Definitions Guide", width="large")
def show_metrics_definitions():
    st.markdown("""
    ## Scale Metrics
    
    **Orders**: Number of orders delivered by couriers
    
    The metric defines the scale and importance of the market to the business.
    
    **Use case**: Where the greatest performance lever is?  
    *Lagging indicator*
    
    ---
    
    **Growth Index**: Weighted average of order growth over the last 12 months
    
    Recent months are weighted more heavily to capture trend changes. This allows the metric to be responsive to recent shifts while still considering longer-term patterns.
    
    **Formula**: Month-over-month growth rates are weighted linearly, with the most recent month having the highest weight (e.g., weights of 1, 2, 3... for oldest to newest).
    
    **Use case**: How is the market trending? Is growth accelerating or decelerating?  
    *Leading indicator* - Can predict future market size and resource needs.
    
    ---
    
    ## Supply Health Metrics
    
    **Delivery CPO**: Cost per order in euros associated with the delivery of the order
    
    **Assumptions:**
    - For simplicity we assume this is a direct variable cost only (promo, courier pay & incentives)
    - This does not include indirect variable costs (insurance, refunds, chargebacks, support costs)
    
    **Use case**: Is the unit economy of delivery reasonable to scale the market or require optimisation?
    
    **Problem with the metric**: The dataset does not provide the revenue side for unit economy calculations, so we can't reasonably make assumptions about economic efficiency of any market. Even higher Delivery CPO may not be a problem if market features higher Average Check and user tolerance towards delivery fees.
    
    *Lagging indicator*
    
    ---
    
    **OPH**: Average number of orders completed by 1 courier per operating hour (orders per hour)
    
    The metric represents courier utilisation. Forecasting demand, Scheduling, Dispatch, and Courier-facing products will all have this as the core optimisation KPI.
    
    **Use case**: How effectively do we utilise the available couriers?
    
    - **High OPH** may be a native feature of a dense urban market but may also be a sign of overutilisation and burnout of the couriers, so should be monitored together with courier churn
    - **Low OPH** would be a feature of less dense markets (large travel distance), but also oversupplied by couriers. It has a critical impact on courier earnings and Full Unit economy efficiency
    
    *Leading indicator*
    
    ---
    
    **Closing**: % of time that the delivery network is closed (i.e. customers cannot place orders) because there aren't enough couriers to fulfill the demand
    
    **Use case**: Do we need to immediately address courier acquisition, prediction, or scheduling issues?
    
    It means the market either grows faster than the courier base or courier supply deteriorates for any reason. May be a sign of failure of the Forecasting demand and Scheduling algorithms, or more broad courier acquisition.
    
    *Leading indicator*
    
    ---
    
    ## User Experience Metrics
    
    **PtoD**: Average time in minutes from customer placing an order to courier arriving at the customer's door (placed to delivered)
    
    **Use case**: Do people need to wait an accaptable time for this market? How do we stack against competition?
    
    This includes merchant preparation time, order pickup time, delivery time. Deterioration of this metric may represent courier supply issues, failure of the algorithm predicting the order pickup time from merchant. Severe issues with PtoD likely to impact user retention.
    
    *Leading indicator*
    
    ---
    
    **UDO**: % of orders that customers claim to not have been delivered and for which a refund is provided
    
    This is a critical metric related to customer trust. Failure to deliver an order may lead to immediate churn.
    
    **Use case**: Do we need to focus on opperational issues or fraud detection?
    
    High UDO may be an indication of undersupply, operational issues with courier onboarding/tracking, or fraud detection gaps.
    
    *Lagging indicator*
    """)

# Button to open metrics definitions dialog
if st.sidebar.button("📚 View Metrics Guide", use_container_width=True):
    show_metrics_definitions()

# Initial filter by date only (country filter moved to sections)
filtered_df = df[
    (df['date_month'].dt.date >= date_range[0]) & 
    (df['date_month'].dt.date <= date_range[1])
]

# Calculate country clusters based on latest period order volume
latest_date_for_clustering = filtered_df['date_month'].max()
latest_orders = filtered_df[
    (filtered_df['date_month'] == latest_date_for_clustering) & 
    (filtered_df['metric'] == 'Orders') &
    (filtered_df['country'] != 'Total')  # Exclude Total from clustering
][['country', 'metric_value']].copy()
latest_orders.columns = ['country', 'orders']

# Apply cluster assignment using the function
latest_orders['cluster'] = latest_orders['orders'].apply(assign_cluster)
country_clusters = dict(zip(latest_orders['country'], latest_orders['cluster']))

# Add Total as a special "country"
country_clusters['Total'] = '🌍 Total'

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
    
    # Calculate % of total orders (excluding Total itself) using the function
    metrics_table['Orders_pct_total'] = calculate_percent_of_total(metrics_table)
    
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
        for metric in ['Orders', 'Growth Index', 'Delivery CPO', 'OPH', 'PtoD', 'Closing', 'UDO']:
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
        'Growth Index': 'Weighted average of last 12 months order growth (recent months weighted higher)',
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
    
    metric_order = ['Orders', 'Growth Index', 'Delivery CPO', 'OPH', 'PtoD', 'Closing', 'UDO']
    
    # Define which metrics are "bad when increased" (reverse coloring)
    bad_when_increased = ['Delivery CPO', 'PtoD', 'Closing', 'UDO']
    
    # Metrics that should not show comparison columns
    no_comparison_metrics = ['Growth Index']
    
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
                elif metric == 'Growth Index':
                    formatted_values.append(f'{val * 100:+.1f}%')
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
            
            # Add % of total after Orders (before Growth Index)
            if metric == 'Orders' and 'Orders_pct_total' in metrics_table.columns:
                # First add MoM comparison for Orders
                if 'Orders' in comparison_data and 'Orders' not in no_comparison_metrics:
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
                    is_bad_metric = 'Orders' in bad_when_increased
                    
                    for comp_val in comparison_data['Orders']:
                        if pd.isna(comp_val) or comp_val is None:
                            comparison_formatted.append('-')
                            comparison_colors.append('white')
                        else:
                            comparison_formatted.append(f'{comp_val:+.1f}%')
                            comparison_colors.append(get_gradient_color(comp_val, is_bad_metric))
                    
                    cell_values.append(comparison_formatted)
                    fill_colors.append(comparison_colors)
                
                # Skip adding comparison column after the metric itself for Orders
                continue
            
            # Add Growth Index right after Orders and its MoM
            if metric == 'Growth Index':
                # Add % of Total after Growth Index
                if 'Orders_pct_total' in metrics_table.columns:
                    header_values.append('<b>% of Total</b>')
                    header_tooltips.append('Country\'s share of total orders in the period')
                    column_widths.append(80)
                    cell_align.append('right')
                    cell_font_sizes.append(10)  # Smaller for secondary metric
                    cell_font_colors.append('#666666')  # Lighter gray for secondary
                    
                    pct_formatted = []
                    for val in metrics_table['Orders_pct_total']:
                        if pd.isna(val):
                            pct_formatted.append('-')
                        else:
                            pct_formatted.append(f'{val:.1f}%')
                    
                    cell_values.append(pct_formatted)
                    fill_colors.append(['white'] * len(metrics_table))
                
                # Don't add comparison column for Growth Index
                continue
            
            # Add comparison column if data exists and metric is not in no_comparison list
            if metric in comparison_data and metric not in no_comparison_metrics:
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
    st.markdown("**Scale & Growth** | **Delivery Ops** | **User Satisfaction**")
    
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
        ['OPH', 'Delivery CPO', 'PtoD', 'Closing', 'UDO', 'Growth Index'],
        index=0,  # Default to OPH
        key="bubble_x_metric"
    )

with bubble_col2:
    y_metric = st.selectbox(
        "Y-Axis Metric",
        ['Growth Index', 'PtoD', 'Delivery CPO', 'OPH', 'Closing', 'UDO'],
        index=0,  # Default to Growth Index
        key="bubble_y_metric"
    )

# Create bubble chart using latest period data
if not latest_period_df.empty:
    # Pivot to get all metrics for bubble chart (excluding Total)
    bubble_data = latest_period_df[latest_period_df['country'] != 'Total'].pivot_table(
        index='country',
        columns='metric',
        values='metric_value',
        aggfunc='mean'
    ).reset_index()
    
    # Add cluster information
    bubble_data['Cluster'] = bubble_data['country'].map(country_clusters)
    
    # Convert percentage metrics for display
    if x_metric in ['Closing', 'UDO', 'Growth Index'] and x_metric in bubble_data.columns:
        bubble_data[f'{x_metric}_display'] = bubble_data[x_metric] * 100
        x_col = f'{x_metric}_display'
    else:
        x_col = x_metric
    
    if y_metric in ['Closing', 'UDO', 'Growth Index'] and y_metric in bubble_data.columns:
        bubble_data[f'{y_metric}_display'] = bubble_data[y_metric] * 100
        y_col = f'{y_metric}_display'
    else:
        y_col = y_metric
    
    # Create bubble chart
    import plotly.graph_objects as go
    
    # Define cluster colors matching the emoji indicators
    cluster_colors = {
        "⭐ Star (50K+)": "#FFD700",        # Gold
        "🔵 L (20K-50K)": "#4472C4",       # Blue
        "🟢 M (5K-20K)": "#70AD47",        # Green
        "🟡 S (<5K)": "#FFC000"            # Orange
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
    x_label = f'{x_metric} {"(%)" if x_metric in ["Closing", "UDO", "Growth Index"] else ""}'
    y_label = f'{y_metric} {"(%)" if y_metric in ["Closing", "UDO", "Growth Index"] else ""}'
    
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
    
    # Add horizontal line at y=0 if Y axis is Growth Index
    if y_metric == 'Growth Index':
        fig_bubble.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray", 
            line_width=2,
            annotation_text="Zero Growth",
            annotation_position="right"
        )
    
    # Add vertical line at x=0 if X axis is Growth Index
    if x_metric == 'Growth Index':
        fig_bubble.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="gray", 
            line_width=2,
            annotation_text="Zero Growth",
            annotation_position="top"
        )
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption(f"Bubble size represents Orders volume • Data for: {latest_date.strftime('%B %Y')}")
else:
    st.info("No data available for bubble chart.")

# 3. METRIC TRENDLINE ANALYSIS - 2x3 Grid
st.subheader("📈 Metric Trendlines - All Metrics View")

# Create filter options
view_type = st.radio(
    "View by:",
    ["Total", "Cluster", "Country"],
    horizontal=True,
    key="trendline_view_type"
)

# Determine which countries to show based on view type
selected_countries_trend = []

if view_type == "Total":
    selected_countries_trend = ['Total']
    st.caption("Showing aggregate metrics across all countries")
    
elif view_type == "Cluster":
    st.markdown("**Select Market Size Cluster:**")
    
    # Get list of clusters (excluding Total)
    available_clusters = sorted([c for c in set(country_clusters.values()) if c != '🌍 Total'])
    
    # Create tile buttons for clusters
    cols = st.columns(len(available_clusters))
    selected_cluster = None
    
    for idx, cluster in enumerate(available_clusters):
        with cols[idx]:
            # Use button style to create tiles
            if st.button(
                cluster,
                key=f"cluster_btn_{idx}",
                use_container_width=True,
                type="primary" if idx == 0 else "secondary"
            ):
                selected_cluster = cluster
    
    # Default to first cluster if none selected
    if selected_cluster is None:
        selected_cluster = available_clusters[0]
    
    # Store selection in session state
    if 'selected_cluster' not in st.session_state:
        st.session_state.selected_cluster = available_clusters[0]
    
    # Update session state when button is clicked
    if selected_cluster:
        st.session_state.selected_cluster = selected_cluster
    
    # Get all countries in the selected cluster
    selected_countries_trend = [
        country for country, cluster in country_clusters.items() 
        if cluster == st.session_state.selected_cluster and country != 'Total'
    ]
    st.caption(f"Showing {len(selected_countries_trend)} countries in {st.session_state.selected_cluster}")
    
else:  # Country view
    st.markdown("**Select Countries:**")
    
    # Get list of countries (excluding Total)
    available_countries = sorted([c for c in country_clusters.keys() if c != 'Total'])
    
    # Initialize session state for country selection
    if 'selected_countries_tiles' not in st.session_state:
        st.session_state.selected_countries_tiles = [available_countries[0]] if available_countries else []
    
    # Create tile buttons for countries (show in rows of 4)
    num_cols = 4
    num_rows = (len(available_countries) + num_cols - 1) // num_cols
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            country_idx = row * num_cols + col_idx
            if country_idx < len(available_countries):
                country = available_countries[country_idx]
                with cols[col_idx]:
                    # Check if country is currently selected
                    is_selected = country in st.session_state.selected_countries_tiles
                    
                    # Create button with different style based on selection
                    if st.button(
                        f"{'✓ ' if is_selected else ''}{country}",
                        key=f"country_btn_{country_idx}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        # Toggle selection
                        if is_selected:
                            st.session_state.selected_countries_tiles.remove(country)
                        else:
                            st.session_state.selected_countries_tiles.append(country)
                        st.rerun()
    
    selected_countries_trend = st.session_state.selected_countries_tiles
    
    if not selected_countries_trend:
        st.caption("⚠️ Please select at least one country")

# Create 2x3 grid of charts
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
    
    # Create 2 rows with 3 columns each (6 metrics total)
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
                            # Show legend on top-right chart, but only if more than one country
                            showlegend=(metric_idx == 2 and len(selected_countries_trend) > 1)
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
