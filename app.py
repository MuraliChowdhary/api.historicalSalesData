# app.py
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
from datetime import datetime, timedelta
import joblib   
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample data structure to work with
sample_data = {
    'inventory': [
        {
            'id': '1',
            'name': 'Tata Salt 1kg',
            'category': 'Cooking Essentials',
            'quantity': 45,
            'threshold': 20,
            'price': 28.0,
            'expirationDate': '2025-06-15',
            'reorderLevel': 25,
            'reorderQuantity': 100,
            'dailyAvgSales': 4.5,
            'lastRestockDate': '2025-03-15',
            'maxCapacity': 200,
            'safetyStock': 15,
            'isDemand': 'HIGH',
            'demandForecast': 5.2,
            'lastDemandUpdate': '2025-04-20',
            'transactions': [
                {'date': '2025-04-23', 'quantity': 5, 'total': 140.0},
                {'date': '2025-04-22', 'quantity': 4, 'total': 112.0},
                {'date': '2025-04-21', 'quantity': 6, 'total': 168.0}
            ]
        },
        {
            'id': '2',
            'name': 'Amul Milk 1L',
            'category': 'Dairy',
            'quantity': 12,
            'threshold': 20,
            'price': 58.0,
            'expirationDate': '2025-04-28',
            'reorderLevel': 15,
            'reorderQuantity': 50,
            'dailyAvgSales': 8.2,
            'lastRestockDate': '2025-04-20',
            'maxCapacity': 100,
            'safetyStock': 10,
            'isDemand': 'HIGH',
            'demandForecast': 9.0,
            'lastDemandUpdate': '2025-04-22',
            'transactions': [
                {'date': '2025-04-23', 'quantity': 10, 'total': 580.0},
                {'date': '2025-04-22', 'quantity': 8, 'total': 464.0},
                {'date': '2025-04-21', 'quantity': 9, 'total': 522.0}
            ]
        },
        {
            'id': '3',
            'name': 'Britannia Biscuit Family Pack',
            'category': 'Snacks',
            'quantity': 18,
            'threshold': 10,
            'price': 45.0,
            'expirationDate': '2025-04-27',
            'reorderLevel': 12,
            'reorderQuantity': 30,
            'dailyAvgSales': 3.5,
            'lastRestockDate': '2025-04-15',
            'maxCapacity': 50,
            'safetyStock': 8,
            'isDemand': 'MEDIUM',
            'demandForecast': 4.0,
            'lastDemandUpdate': '2025-04-20',
            'transactions': [
                {'date': '2025-04-23', 'quantity': 4, 'total': 180.0},
                {'date': '2025-04-22', 'quantity': 3, 'total': 135.0},
                {'date': '2025-04-21', 'quantity': 4, 'total': 180.0}
            ]
        },
        {
            'id': '4',
            'name': 'Tata Rice 5kg',
            'category': 'Cooking Essentials',
            'quantity': 25,
            'threshold': 15,
            'price': 550.0,
            'expirationDate': '2025-10-15',
            'reorderLevel': 20,
            'reorderQuantity': 25,
            'dailyAvgSales': 2.3,
            'lastRestockDate': '2025-04-10',
            'maxCapacity': 50,
            'safetyStock': 12,
            'isDemand': 'MEDIUM',
            'demandForecast': 2.5,
            'lastDemandUpdate': '2025-04-20',
            'transactions': [
                {'date': '2025-04-23', 'quantity': 3, 'total': 1650.0},
                {'date': '2025-04-22', 'quantity': 2, 'total': 1100.0},
                {'date': '2025-04-21', 'quantity': 2, 'total': 1100.0}
            ]
        },
        {
            'id': '5',
            'name': 'Colgate Toothpaste 200g',
            'category': 'Personal Care',
            'quantity': 30,
            'threshold': 15,
            'price': 95.0,
            'expirationDate': '2026-01-15',
            'reorderLevel': 20,
            'reorderQuantity': 40,
            'dailyAvgSales': 3.8,
            'lastRestockDate': '2025-04-05',
            'maxCapacity': 60,
            'safetyStock': 10,
            'isDemand': 'LOW',
            'demandForecast': 3.5,
            'lastDemandUpdate': '2025-04-18',
            'transactions': [
                {'date': '2025-04-23', 'quantity': 4, 'total': 380.0},
                {'date': '2025-04-22', 'quantity': 3, 'total': 285.0},
                {'date': '2025-04-21', 'quantity': 5, 'total': 475.0}
            ]
        }
    ]
}

# Sample sales data
sample_sales = [
    {'date': '2025-04-23', 'totalSales': 2930.0, 'totalItems': 26, 'bills': 18},
    {'date': '2025-04-22', 'totalSales': 2096.0, 'totalItems': 20, 'bills': 15},
    {'date': '2025-04-21', 'totalSales': 2445.0, 'totalItems': 26, 'bills': 17},
    {'date': '2025-04-20', 'totalSales': 3105.0, 'totalItems': 32, 'bills': 20},
    {'date': '2025-04-19', 'totalSales': 3450.0, 'totalItems': 35, 'bills': 22},
    {'date': '2025-04-18', 'totalSales': 2780.0, 'totalItems': 28, 'bills': 19},
    {'date': '2025-04-17', 'totalSales': 2565.0, 'totalItems': 25, 'bills': 16},
    {'date': '2025-04-16', 'totalSales': 2875.0, 'totalItems': 30, 'bills': 18},
    {'date': '2025-04-15', 'totalSales': 3220.0, 'totalItems': 33, 'bills': 21},
    {'date': '2025-04-14', 'totalSales': 2450.0, 'totalItems': 24, 'bills': 17},
    {'date': '2025-04-13', 'totalSales': 2680.0, 'totalItems': 27, 'bills': 18},
    {'date': '2025-04-12', 'totalSales': 3380.0, 'totalItems': 34, 'bills': 23},
    {'date': '2025-04-11', 'totalSales': 2980.0, 'totalItems': 30, 'bills': 20},
    {'date': '2025-04-10', 'totalSales': 2760.0, 'totalItems': 28, 'bills': 19},
    {'date': '2025-04-09', 'totalSales': 3120.0, 'totalItems': 32, 'bills': 21},
    {'date': '2025-04-08', 'totalSales': 2890.0, 'totalItems': 29, 'bills': 19},
    {'date': '2025-04-07', 'totalSales': 3250.0, 'totalItems': 33, 'bills': 22},
    {'date': '2025-04-06', 'totalSales': 2550.0, 'totalItems': 26, 'bills': 17},
    {'date': '2025-04-05', 'totalSales': 2970.0, 'totalItems': 30, 'bills': 20},
    {'date': '2025-04-04', 'totalSales': 3340.0, 'totalItems': 34, 'bills': 22},
    {'date': '2025-04-03', 'totalSales': 2830.0, 'totalItems': 29, 'bills': 19},
    {'date': '2025-04-02', 'totalSales': 2670.0, 'totalItems': 27, 'bills': 18},
    {'date': '2025-04-01', 'totalSales': 3080.0, 'totalItems': 31, 'bills': 20},
    {'date': '2025-03-31', 'totalSales': 2620.0, 'totalItems': 27, 'bills': 18},
    {'date': '2025-03-30', 'totalSales': 2950.0, 'totalItems': 30, 'bills': 19},
    {'date': '2025-03-29', 'totalSales': 3180.0, 'totalItems': 32, 'bills': 21},
    {'date': '2025-03-28', 'totalSales': 2730.0, 'totalItems': 28, 'bills': 18},
    {'date': '2025-03-27', 'totalSales': 2500.0, 'totalItems': 25, 'bills': 17},
    {'date': '2025-03-26', 'totalSales': 2870.0, 'totalItems': 29, 'bills': 19},
    {'date': '2025-03-25', 'totalSales': 3410.0, 'totalItems': 35, 'bills': 23}
]




# Function to convert data to DataFrame
def get_inventory_df():
    # In a real application, this would pull from your database
    df = pd.DataFrame(sample_data['inventory'])
    # Parse dates
    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    df['lastRestockDate'] = pd.to_datetime(df['lastRestockDate'])
    df['lastDemandUpdate'] = pd.to_datetime(df['lastDemandUpdate'])
    return df

def get_sales_df():
    # In a real application, this would pull from your database
    df = pd.DataFrame(sample_sales)
    df['date'] = pd.to_datetime(df['date'])
    return df

@app.route('/api/ml/reorder-recommendations', methods=['GET'])
def reorder_recommendations():
    """Provides reorder recommendations based on inventory levels and demand"""
    inventory_df = get_inventory_df()
    
    # Calculate days until reorder needed
    today = datetime.now()
    
    recommendations = []
    for _, item in inventory_df.iterrows():
        days_of_supply = 0
        if item['dailyAvgSales'] > 0:
            days_of_supply = (item['quantity'] - item['safetyStock']) / item['dailyAvgSales']
        
        days_until_expiry = None
        if pd.notnull(item['expirationDate']):
            days_until_expiry = (item['expirationDate'] - today).days
        
        urgency = 'LOW'
        if days_of_supply <= 3:
            urgency = 'HIGH'
        elif days_of_supply <= 7:
            urgency = 'MEDIUM'
        
        # Check if current quantity is below reorder level
        needs_reorder = item['quantity'] <= item['reorderLevel']
        
        if needs_reorder:
            recommendations.append({
                'id': item['id'],
                'name': item['name'],
                'currentStock': item['quantity'],
                'reorderLevel': item['reorderLevel'],
                'recommendedOrderQuantity': item['reorderQuantity'],
                'daysOfSupplyRemaining': round(days_of_supply, 1),
                'daysUntilExpiry': days_until_expiry,
                'urgency': urgency,
                'reason': 'Stock below reorder level'
            })
    
    # Sort by urgency (HIGH > MEDIUM > LOW)
    urgency_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: urgency_rank[x['urgency']])
    
    return jsonify({
        'count': len(recommendations),
        'recommendations': recommendations
    })

@app.route('/api/ml/summary', methods=['GET'])
def get_summary():
    """Provides a daily summary report with key metrics"""
    inventory_df = get_inventory_df()
    sales_df = get_sales_df()
    
    # Get today's sales (or most recent date in our sample data)
    today = sales_df['date'].max()
    today_sales = sales_df[sales_df['date'] == today].iloc[0]
    
    # Get yesterday's sales
    yesterday = today - timedelta(days=1)
    yesterday_sales = sales_df[sales_df['date'] == yesterday].iloc[0]
    
    # Calculate day-over-day change
    sales_change_pct = ((today_sales['totalSales'] - yesterday_sales['totalSales']) / 
                         yesterday_sales['totalSales'] * 100)
    
    # Low stock items
    low_stock_items = inventory_df[inventory_df['quantity'] <= inventory_df['threshold']].shape[0]
    
    # Items expiring soon (within 7 days)
    expiring_soon = inventory_df[
        (inventory_df['expirationDate'] - today).dt.days <= 7
    ].shape[0]
    
    # Calculate total inventory value
    total_inventory_value = (inventory_df['quantity'] * inventory_df['price']).sum()
    
    # Get last 7 days sales
    last_7_days = sales_df[sales_df['date'] >= (today - timedelta(days=6))]
    weekly_sales = last_7_days['totalSales'].sum()
    
    # Calculate inventory turnover rate (annualized)
    if weekly_sales > 0:
        weekly_turnover = weekly_sales / total_inventory_value
        annual_turnover = weekly_turnover * 52  # Annualize
    else:
        annual_turnover = 0
        
    # Calculate average basket size
    avg_basket_size = today_sales['totalSales'] / today_sales['bills'] if today_sales['bills'] > 0 else 0
    
    return jsonify({
        'date': today.strftime('%Y-%m-%d'),
        'dailySales': {
            'total': float(today_sales['totalSales']),
            'items': int(today_sales['totalItems']),
            'transactions': int(today_sales['bills']),
            'averageBasketSize': round(avg_basket_size, 2),
            'changeFromYesterday': round(sales_change_pct, 2)
        },
        'inventory': {
            'totalValue': round(total_inventory_value, 2),
            'lowStockItems': int(low_stock_items),
            'expiringWithin7Days': int(expiring_soon),
            'turnoverRate': round(annual_turnover, 2)
        },
        'weeklySales': round(weekly_sales, 2),
        'monthlySales': round(weekly_sales * 4.33, 2)  # Approximation
    })

@app.route('/api/ml/top-performing-products', methods=['GET'])
def top_performing_products():
    """Returns the top performing products based on sales volume and profit margin"""
    inventory_df = get_inventory_df()
    
    # In a real application, we would calculate this from actual transaction data
    # For this example, we'll simulate it based on dailyAvgSales
    
    # Calculate estimated monthly sales and revenue for each product
    analysis_results = []
    for _, item in inventory_df.iterrows():
        monthly_units = item['dailyAvgSales'] * 30
        monthly_revenue = monthly_units * item['price']
        
        # Simulating cost data (normally this would come from your database)
        # Assuming profit margin between 15-40% depending on category
        if item['category'] == 'Dairy':
            profit_margin = 0.15  # Lower margins on perishables
        elif item['category'] == 'Personal Care':
            profit_margin = 0.40  # Higher margins on personal care
        else:
            profit_margin = 0.25  # Average margin for other products
            
        monthly_profit = monthly_revenue * profit_margin
        
        analysis_results.append({
            'id': item['id'],
            'name': item['name'],
            'category': item['category'],
            'price': float(item['price']),
            'dailyAvgSales': float(item['dailyAvgSales']),
            'monthlyUnitsSold': round(monthly_units, 1),
            'monthlyRevenue': round(monthly_revenue, 2),
            'estimatedProfitMargin': round(profit_margin * 100, 1),
            'monthlyProfit': round(monthly_profit, 2),
            'demand': item['isDemand']
        })
    
    # Sort by monthly revenue (highest first)
    analysis_results.sort(key=lambda x: x['monthlyRevenue'], reverse=True)
    
    return jsonify({
        'topProducts': analysis_results[:5],  # Top 5 products
        'totalProducts': len(analysis_results),
        'totalMonthlyRevenue': round(sum(item['monthlyRevenue'] for item in analysis_results), 2),
        'averageProfitMargin': round(sum(item['estimatedProfitMargin'] for item in analysis_results) / len(analysis_results), 1)
    })

@app.route('/api/ml/high-demand-items', methods=['GET'])
def high_demand_items():
    """Identifies products with high demand or significant changes in demand"""
    inventory_df = get_inventory_df()
    
    high_demand_products = []
    for _, item in inventory_df.iterrows():
        # Calculate stock days remaining
        days_remaining = item['quantity'] / item['dailyAvgSales'] if item['dailyAvgSales'] > 0 else 30
        
        # Check if actual demand exceeds forecasted demand
        demand_vs_forecast = 0
        if item['demandForecast'] > 0:
            demand_vs_forecast = (item['dailyAvgSales'] / item['demandForecast'] - 1) * 100
            
        # Calculate inventory turnover (daily sales / average inventory)
        inventory_turnover = item['dailyAvgSales'] / ((item['quantity'] + item['reorderLevel']) / 2)
        
        if item['isDemand'] == 'HIGH' or demand_vs_forecast > 15:
            high_demand_products.append({
                'id': item['id'],
                'name': item['name'],
                'category': item['category'],
                'currentStock': int(item['quantity']),
                'dailyAvgSales': float(item['dailyAvgSales']),
                'daysRemaining': round(days_remaining, 1),
                'demandTrend': item['isDemand'],
                'demandVsForecast': round(demand_vs_forecast, 1),
                'turnoverRate': round(inventory_turnover * 30, 2),  # Monthly turnover
                'reorderSuggestion': item['reorderQuantity'] if days_remaining < 7 else 0
            })
    
    # Sort by days remaining (ascending)
    high_demand_products.sort(key=lambda x: x['daysRemaining'])
    
    return jsonify({
        'count': len(high_demand_products),
        'highDemandItems': high_demand_products,
        'analysisDate': datetime.now().strftime('%Y-%m-%d')
    })

@app.route('/api/ml/profit-margin-analysis', methods=['GET'])
def profit_margin_analysis():
    """Analyzes profit margins by category and product"""
    inventory_df = get_inventory_df()
    
    # Simulate cost data by category (in a real application, this would come from your database)
    category_margins = {
        'Cooking Essentials': 0.25,
        'Dairy': 0.15,
        'Snacks': 0.35,
        'Personal Care': 0.40,
        'Beverages': 0.30
    }
    
    # Calculate profit margins for products and categories
    product_margins = []
    category_data = {}
    
    for _, item in inventory_df.iterrows():
        category = item['category']
        profit_margin = category_margins.get(category, 0.25)  # Default 25% if category not found
        
        # Calculate cost price
        cost_price = item['price'] * (1 - profit_margin)
        
        # Calculate monthly revenue and profit
        monthly_units = item['dailyAvgSales'] * 30
        monthly_revenue = monthly_units * item['price']
        monthly_profit = monthly_revenue * profit_margin
        
        # Add to product margins list
        product_margins.append({
            'id': item['id'],
            'name': item['name'],
            'category': category,
            'price': float(item['price']),
            'costPrice': round(cost_price, 2),
            'profitMargin': round(profit_margin * 100, 1),
            'monthlyRevenue': round(monthly_revenue, 2),
            'monthlyProfit': round(monthly_profit, 2),
            'contribution': 0  # Will calculate after getting totals
        })
        
        # Aggregate by category
        if category not in category_data:
            category_data[category] = {
                'totalRevenue': 0,
                'totalProfit': 0,
                'averageMargin': 0,
                'productCount': 0
            }
        
        category_data[category]['totalRevenue'] += monthly_revenue
        category_data[category]['totalProfit'] += monthly_profit
        category_data[category]['productCount'] += 1
    
    # Calculate totals
    total_monthly_profit = sum(item['monthlyProfit'] for item in product_margins)
    
    # Calculate profit contribution percentage for each product
    for item in product_margins:
        if total_monthly_profit > 0:
            item['contribution'] = round(item['monthlyProfit'] / total_monthly_profit * 100, 1)
    
    # Calculate average margins for categories
    category_summary = []
    for category, data in category_data.items():
        if data['productCount'] > 0:
            data['averageMargin'] = round(data['totalProfit'] / data['totalRevenue'] * 100, 1)
            category_summary.append({
                'category': category,
                'productCount': data['productCount'],
                'totalRevenue': round(data['totalRevenue'], 2),
                'totalProfit': round(data['totalProfit'], 2),
                'averageMargin': data['averageMargin'],
                'contribution': round(data['totalProfit'] / total_monthly_profit * 100, 1) if total_monthly_profit > 0 else 0
            })
    
    # Sort products by profit contribution (highest first)
    product_margins.sort(key=lambda x: x['contribution'], reverse=True)
    
    # Sort categories by profit contribution (highest first)
    category_summary.sort(key=lambda x: x['contribution'], reverse=True)
    
    return jsonify({
        'totalMonthlyProfit': round(total_monthly_profit, 2),
        'overallMargin': round(total_monthly_profit / sum(item['monthlyRevenue'] for item in product_margins) * 100, 1),
        'categoryAnalysis': category_summary,
        'topProfitProducts': product_margins[:5],
        'lowMarginProducts': sorted(product_margins, key=lambda x: x['profitMargin'])[:3]
    })

@app.route('/api/ml/seasonal-analysis', methods=['GET'])
def seasonal_analysis():
    """Analyzes seasonal patterns in product demand"""
    # Create a simple seasonal model for demonstration
    # In a real application, this would use historical data and actual ML models
    
    now = datetime.now()
    current_month = now.month
    
    seasonal_patterns = {
        'Cooking Essentials': {
            'peak_months': [1, 5, 10, 11, 12],  # Festive seasons and summer
            'low_months': [2, 3, 7, 8],
            'pattern': 'Peaks during festivals and summer'
        },
        'Dairy': {
            'peak_months': [4, 5, 6, 7],  # Summer months
            'low_months': [1, 2, 11, 12],
            'pattern': 'Higher demand in summer'
        },
        'Snacks': {
            'peak_months': [1, 5, 6, 10, 11, 12],  # School holidays and festivals
            'low_months': [2, 3, 8, 9],
            'pattern': 'Peaks during holidays and festivals'
        },
        'Personal Care': {
            'peak_months': [5, 6, 7, 8],  # Summer months
            'low_months': [1, 2, 3],
            'pattern': 'Higher in summer months'
        },
        'Beverages': {
            'peak_months': [3, 4, 5, 6, 7, 8],  # Summer months
            'low_months': [11, 12, 1, 2],
            'pattern': 'Higher in summer, lower in winter'
        }
    }
    
    inventory_df = get_inventory_df()
    
    seasonal_insights = []
    upcoming_peaks = []
    
    for _, item in inventory_df.iterrows():
        category = item['category']
        if category in seasonal_patterns:
            pattern = seasonal_patterns[category]
            
            # Determine current seasonal status
            seasonal_status = 'NORMAL'
            if current_month in pattern['peak_months']:
                seasonal_status = 'PEAK'
            elif current_month in pattern['low_months']:
                seasonal_status = 'LOW'
            
            # Find next peak month
            next_peak = None
            months_away = 12  # Maximum possible
            
            for peak_month in pattern['peak_months']:
                if peak_month > current_month:
                    months_until_peak = peak_month - current_month
                    if months_until_peak < months_away:
                        months_away = months_until_peak
                        next_peak = peak_month
            
            # If no future peak found in this year, check first peak of next year
            if next_peak is None and pattern['peak_months']:
                next_peak = pattern['peak_months'][0]
                months_away = 12 - current_month + next_peak
            
            # Create seasonal insight for this product
            insight = {
                'id': item['id'],
                'name': item['name'],
                'category': category,
                'currentStatus': seasonal_status,
                'seasonalPattern': pattern['pattern'],
                'nextPeakMonth': next_peak,
                'monthsUntilNextPeak': months_away,
                'recommendedAction': 'STOCK_UP' if months_away <= 1 and next_peak is not None else 'MONITOR'
            }
            
            seasonal_insights.append(insight)
            
            # Add to upcoming peaks if within 2 months
            if months_away <= 2 and next_peak is not None:
                upcoming_peaks.append({
                    'id': item['id'],
                    'name': item['name'],
                    'category': category,
                    'peakMonth': next_peak,
                    'monthsAway': months_away,
                    'suggestedStockIncrease': '30-50%' if seasonal_status != 'PEAK' else '10-20%'
                })
    
    # Sort seasonal insights by months until next peak
    seasonal_insights.sort(key=lambda x: x['monthsUntilNextPeak'])
    
    # Sort upcoming peaks by months away
    upcoming_peaks.sort(key=lambda x: x['monthsAway'])
    
    # Get categories currently in peak season
    current_peak_categories = [cat for cat, pattern in seasonal_patterns.items() 
                              if current_month in pattern['peak_months']]
    
    return jsonify({
        'currentMonth': current_month,
        'currentPeakCategories': current_peak_categories,
        'seasonalInsights': seasonal_insights,
        'upcomingPeakProducts': upcoming_peaks,
        'analysisDate': now.strftime('%Y-%m-%d')
    })



if __name__ == '__main__':
    app.run(debug=True, port=8000)
