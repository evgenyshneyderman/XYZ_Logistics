"""
Unit tests for dashboard metrics calculations.

Tests the ACTUAL functions from app.py to ensure correct calculations.

Tests cover:
- Growth Index calculation (weighted average with linear weights)
- Total aggregations (Orders sum, CPO/PtoD/UDO weighted averages)
- Cluster assignments
- % of Total calculations

Run with: pytest test_metrics.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import actual functions from app.py
# Note: We need to mock streamlit's cache decorator since we're testing outside Streamlit
import streamlit as st

# Mock the cache_data decorator for testing
if not hasattr(st, 'cache_data'):
    def cache_data(func):
        return func
    st.cache_data = cache_data

# Now import the actual functions from app.py
from app import calculate_weighted_growth, calculate_totals, assign_cluster, calculate_percent_of_total


class TestGrowthIndexCalculation:
    """Test Growth Index weighted average calculation using ACTUAL app.py function"""
    
    def test_growth_index_simple_growth(self):
        """Test Growth Index with consistent 10% growth"""
        # Create test data: 3 countries with consistent growth
        dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        data = []
        
        for date in dates:
            data.extend([
                {'country': 'Country_A', 'date_month': date, 'metric': 'Orders', 'metric_value': 100 * (1.1 ** list(dates).index(date))},
            ])
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        
        # Check Growth Index was calculated
        growth_data = df[df['metric'] == 'Growth Index']
        assert len(growth_data) > 0, "Growth Index should be calculated"
        
        # For consistent 10% growth, Growth Index should be close to 0.10
        latest_growth = growth_data[growth_data['date_month'] == growth_data['date_month'].max()]['metric_value'].values[0]
        assert abs(latest_growth - 0.10) < 0.01, f"Expected ~0.10, got {latest_growth}"
    
    def test_growth_index_accelerating_growth(self):
        """Test Growth Index with accelerating growth (recent months higher)"""
        dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        # Orders with accelerating growth: 5%, 10%, 15%
        orders = [100, 105, 115.5, 132.825]
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'country': 'Country_A',
                'date_month': date,
                'metric': 'Orders',
                'metric_value': orders[i]
            })
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        
        growth_data = df[df['metric'] == 'Growth Index']
        latest_growth = growth_data[growth_data['date_month'] == growth_data['date_month'].max()]['metric_value'].values[0]
        
        # Expected: weighted avg with weights [1, 2, 3] = (5*1 + 10*2 + 15*3) / 6 = 11.67%
        expected = 0.1167
        assert abs(latest_growth - expected) < 0.01, f"Expected ~{expected}, got {latest_growth}"
    
    def test_growth_index_negative_growth(self):
        """Test Growth Index with declining orders"""
        dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        # Orders declining by 5% each month
        orders = [100, 95, 90.25, 85.7375]
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'country': 'Country_A',
                'date_month': date,
                'metric': 'Orders',
                'metric_value': orders[i]
            })
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        
        growth_data = df[df['metric'] == 'Growth Index']
        latest_growth = growth_data[growth_data['date_month'] == growth_data['date_month'].max()]['metric_value'].values[0]
        
        # Should be negative
        assert latest_growth < 0, f"Expected negative growth, got {latest_growth}"
        assert abs(latest_growth - (-0.05)) < 0.01, f"Expected ~-0.05, got {latest_growth}"


class TestTotalAggregations:
    """Test Total (aggregate) metric calculations using ACTUAL app.py function"""
    
    def test_total_orders_sum(self):
        """Test that Total orders = sum of all country orders"""
        dates = pd.date_range('2024-01-01', periods=2, freq='MS')
        
        data = []
        for date in dates:
            data.extend([
                {'country': 'Country_A', 'date_month': date, 'metric': 'Orders', 'metric_value': 10000},
                {'country': 'Country_B', 'date_month': date, 'metric': 'Orders', 'metric_value': 20000},
                {'country': 'Country_C', 'date_month': date, 'metric': 'Orders', 'metric_value': 30000},
            ])
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)  # Need this first for Growth Index
        df = calculate_totals(df)
        
        # Check Total was calculated
        total_data = df[(df['country'] == 'Total') & (df['metric'] == 'Orders')]
        assert len(total_data) > 0, "Total Orders should be calculated"
        
        # Verify sum
        for date in dates:
            total_orders = df[(df['country'] == 'Total') & (df['metric'] == 'Orders') & (df['date_month'] == date)]['metric_value'].values[0]
            assert total_orders == 60000, f"Expected 60000, got {total_orders}"
    
    def test_delivery_cpo_weighted_average(self):
        """Test Delivery CPO weighted average calculation"""
        dates = pd.date_range('2024-01-01', periods=1, freq='MS')
        
        data = [
            {'country': 'Country_A', 'date_month': dates[0], 'metric': 'Orders', 'metric_value': 10000},
            {'country': 'Country_B', 'date_month': dates[0], 'metric': 'Orders', 'metric_value': 20000},
            {'country': 'Country_C', 'date_month': dates[0], 'metric': 'Orders', 'metric_value': 30000},
            {'country': 'Country_A', 'date_month': dates[0], 'metric': 'Delivery CPO', 'metric_value': 5.0},
            {'country': 'Country_B', 'date_month': dates[0], 'metric': 'Delivery CPO', 'metric_value': 7.0},
            {'country': 'Country_C', 'date_month': dates[0], 'metric': 'Delivery CPO', 'metric_value': 6.0},
        ]
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        df = calculate_totals(df)
        
        # Get Total CPO
        total_cpo = df[(df['country'] == 'Total') & (df['metric'] == 'Delivery CPO')]['metric_value'].values[0]
        
        # Expected: (5*10000 + 7*20000 + 6*30000) / 60000 = 6.1667
        expected = 6.1667
        assert abs(total_cpo - expected) < 0.001, f"Expected ~{expected}, got {total_cpo}"
    
    def test_total_growth_index_from_total_orders(self):
        """Test that Total Growth Index is calculated from Total orders, not country averages"""
        dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        
        # Create data where countries have consistent growth
        data = []
        for i, date in enumerate(dates):
            # Country A and B both growing at 10%
            data.extend([
                {'country': 'Country_A', 'date_month': date, 'metric': 'Orders', 'metric_value': 50000 * (1.10 ** i)},
                {'country': 'Country_B', 'date_month': date, 'metric': 'Orders', 'metric_value': 50000 * (1.10 ** i)},
            ])
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        df = calculate_totals(df)
        
        # Check that Total Growth Index exists
        total_growth = df[(df['country'] == 'Total') & (df['metric'] == 'Growth Index')]
        assert len(total_growth) > 0, "Total Growth Index should be calculated"
        
        # The Total should show consistent 10% growth since both countries grow at 10%
        latest_growth = total_growth[total_growth['date_month'] == total_growth['date_month'].max()]['metric_value'].values[0]
        assert abs(latest_growth - 0.10) < 0.01, f"Expected ~0.10, got {latest_growth}"


class TestClusterAssignment:
    """Test market size cluster assignment logic using ACTUAL app.py function"""
    
    def test_star_cluster(self):
        """Test Star cluster assignment (50K+)"""
        assert assign_cluster(50000) == "⭐ Star (50K+)"
        assert assign_cluster(100000) == "⭐ Star (50K+)"
    
    def test_l_cluster(self):
        """Test L cluster assignment (20K-50K)"""
        assert assign_cluster(20000) == "📦 L (20K-50K)"
        assert assign_cluster(35000) == "📦 L (20K-50K)"
        assert assign_cluster(49999) == "📦 L (20K-50K)"
    
    def test_m_cluster(self):
        """Test M cluster assignment (5K-20K)"""
        assert assign_cluster(5000) == "📊 M (5K-20K)"
        assert assign_cluster(12000) == "📊 M (5K-20K)"
        assert assign_cluster(19999) == "📊 M (5K-20K)"
    
    def test_s_cluster(self):
        """Test S cluster assignment (<5K)"""
        assert assign_cluster(4999) == "📌 S (<5K)"
        assert assign_cluster(1000) == "📌 S (<5K)"
        assert assign_cluster(100) == "📌 S (<5K)"
    
    def test_boundary_values(self):
        """Test cluster boundaries are correct"""
        # Just below thresholds should go to lower cluster
        assert assign_cluster(49999) == "📦 L (20K-50K)"
        assert assign_cluster(19999) == "📊 M (5K-20K)"
        assert assign_cluster(4999) == "📌 S (<5K)"
        
        # At threshold should go to that cluster
        assert assign_cluster(50000) == "⭐ Star (50K+)"
        assert assign_cluster(20000) == "📦 L (20K-50K)"
        assert assign_cluster(5000) == "📊 M (5K-20K)"


class TestPercentOfTotal:
    """Test % of Total calculation using ACTUAL app.py function"""
    
    def test_percent_of_total_simple(self):
        """Test % of Total calculation with simple numbers"""
        data = {
            'country': ['Country_A', 'Country_B', 'Country_C'],
            'Orders': [10000, 20000, 30000]
        }
        df = pd.DataFrame(data)
        
        pct_total = calculate_percent_of_total(df)
        
        # Expected: 16.67%, 33.33%, 50%
        assert abs(pct_total.iloc[0] - 16.667) < 0.01
        assert abs(pct_total.iloc[1] - 33.333) < 0.01
        assert abs(pct_total.iloc[2] - 50.0) < 0.01
        
        # Sum should be 100%
        assert abs(pct_total.sum() - 100.0) < 0.01
    
    def test_percent_of_total_excludes_total(self):
        """Test that Total row is excluded from % of Total calculation"""
        data = {
            'country': ['Country_A', 'Country_B', 'Total'],
            'Orders': [40000, 60000, 100000]
        }
        df = pd.DataFrame(data)
        
        pct_total = calculate_percent_of_total(df)
        
        # Country_A: 40%, Country_B: 60%, Total: None
        assert abs(pct_total.iloc[0] - 40.0) < 0.01
        assert abs(pct_total.iloc[1] - 60.0) < 0.01
        assert pd.isna(pct_total.iloc[2])
        
        # Sum of non-null values should be 100%
        assert abs(pct_total.dropna().sum() - 100.0) < 0.01
    
    def test_percent_of_total_zero_orders(self):
        """Test % of Total when all orders are zero"""
        data = {
            'country': ['Country_A', 'Country_B'],
            'Orders': [0, 0]
        }
        df = pd.DataFrame(data)
        
        pct_total = calculate_percent_of_total(df)
        
        # Should return None for all when total is zero
        assert pd.isna(pct_total.iloc[0])
        assert pd.isna(pct_total.iloc[1])


class TestEdgeCases:
    """Test edge cases and error handling with actual functions"""
    
    def test_no_closing_or_oph_in_totals(self):
        """Verify that Closing and OPH are NOT calculated for Total"""
        dates = pd.date_range('2024-01-01', periods=1, freq='MS')
        
        data = [
            {'country': 'Country_A', 'date_month': dates[0], 'metric': 'Orders', 'metric_value': 10000},
            {'country': 'Country_A', 'date_month': dates[0], 'metric': 'Closing', 'metric_value': 0.05},
            {'country': 'Country_A', 'date_month': dates[0], 'metric': 'OPH', 'metric_value': 1.5},
        ]
        
        df = pd.DataFrame(data)
        df = calculate_weighted_growth(df)
        df = calculate_totals(df)
        
        total_metrics = df[df['country'] == 'Total']['metric'].unique()
        
        # Should NOT have Closing or OPH
        assert 'Closing' not in total_metrics, "Closing should not be calculated for Total"
        assert 'OPH' not in total_metrics, "OPH should not be calculated for Total"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
