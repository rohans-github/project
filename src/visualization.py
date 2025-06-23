"""
Data Visualization Module
Creates interactive charts and graphs for financial data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FinanceVisualizer:
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'expenses': '#e74c3c',
            'income': '#27ae60',
            'savings': '#3498db'
        }
        
    def create_spending_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create spending overview chart"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            # Separate income and expenses
            expenses = df[df['amount'] < 0].copy()
            income = df[df['amount'] > 0].copy()
            
            expenses['amount'] = expenses['amount'].abs()
            
            # Group by month
            expenses['month'] = expenses['date'].dt.to_period('M')
            income['month'] = income['date'].dt.to_period('M')
            
            monthly_expenses = expenses.groupby('month')['amount'].sum()
            monthly_income = income.groupby('month')['amount'].sum()
            
            # Create figure
            fig = go.Figure()
            
            # Add expenses
            fig.add_trace(go.Bar(
                x=[str(m) for m in monthly_expenses.index],
                y=monthly_expenses.values,
                name='Expenses',
                marker_color=self.color_scheme['expenses'],
                opacity=0.8
            ))
            
            # Add income
            fig.add_trace(go.Bar(
                x=[str(m) for m in monthly_income.index],
                y=monthly_income.values,
                name='Income',
                marker_color=self.color_scheme['income'],
                opacity=0.8
            ))
            
            # Add net savings line
            all_months = sorted(set(monthly_expenses.index) | set(monthly_income.index))
            net_savings = []
            
            for month in all_months:
                month_income = monthly_income.get(month, 0)
                month_expenses = monthly_expenses.get(month, 0)
                net_savings.append(month_income - month_expenses)
            
            fig.add_trace(go.Scatter(
                x=[str(m) for m in all_months],
                y=net_savings,
                mode='lines+markers',
                name='Net Savings',
                line=dict(color=self.color_scheme['savings'], width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='Monthly Income vs Expenses',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                barmode='group',
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating spending chart: {str(e)}")
            return None
    
    def create_category_pie_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create category breakdown pie chart"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            
            # Focus on expenses only
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Group by category
            if 'category' not in expenses.columns:
                # Auto-categorize if not available
                expenses['category'] = 'Other'
            
            category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            # Combine small categories into "Other"
            threshold = category_totals.sum() * 0.05  # 5% threshold
            large_categories = category_totals[category_totals >= threshold]
            small_categories_sum = category_totals[category_totals < threshold].sum()
            
            if small_categories_sum > 0:
                large_categories['Other (Small)'] = small_categories_sum
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=large_categories.index,
                values=large_categories.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto',
                marker=dict(
                    colors=px.colors.qualitative.Set3[:len(large_categories)]
                )
            )])
            
            fig.update_layout(
                title='Spending by Category',
                height=400,
                showlegend=True,
                annotations=[dict(text='Expenses', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating category pie chart: {str(e)}")
            return None
    
    def create_trend_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create spending trend chart over time"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Daily spending
            daily_spending = expenses.groupby(expenses['date'].dt.date)['amount'].sum()
            
            # Calculate moving average
            window = min(7, len(daily_spending) // 3)  # 7-day moving average or 1/3 of data
            if window > 1:
                moving_avg = daily_spending.rolling(window=window, center=True).mean()
            else:
                moving_avg = daily_spending
            
            fig = go.Figure()
            
            # Daily spending bars
            fig.add_trace(go.Bar(
                x=daily_spending.index,
                y=daily_spending.values,
                name='Daily Spending',
                marker_color=self.color_scheme['expenses'],
                opacity=0.6
            ))
            
            # Moving average line
            fig.add_trace(go.Scatter(
                x=moving_avg.index,
                y=moving_avg.values,
                mode='lines',
                name=f'{window}-Day Average',
                line=dict(color=self.color_scheme['primary'], width=3)
            ))
            
            fig.update_layout(
                title='Daily Spending Trend',
                xaxis_title='Date',
                yaxis_title='Amount ($)',
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trend chart: {str(e)}")
            return None
    
    def create_budget_vs_actual_chart(self, transactions: List[Dict], budget: Dict[str, float]) -> Optional[go.Figure]:
        """Create budget vs actual spending comparison"""
        try:
            if not transactions or not budget:
                return None
                
            df = pd.DataFrame(transactions)
            expenses = df[df['amount'] < 0].copy()
            
            if expenses.empty:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Calculate actual spending by category
            if 'category' not in expenses.columns:
                return None
                
            actual_spending = expenses.groupby('category')['amount'].sum()
            
            # Prepare data for chart
            categories = list(set(budget.keys()) | set(actual_spending.index))
            budget_values = [budget.get(cat, 0) for cat in categories]
            actual_values = [actual_spending.get(cat, 0) for cat in categories]
            
            fig = go.Figure()
            
            # Budget bars
            fig.add_trace(go.Bar(
                x=categories,
                y=budget_values,
                name='Budget',
                marker_color=self.color_scheme['info'],
                opacity=0.7
            ))
            
            # Actual spending bars
            fig.add_trace(go.Bar(
                x=categories,
                y=actual_values,
                name='Actual',
                marker_color=self.color_scheme['expenses'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Budget vs Actual Spending',
                xaxis_title='Category',
                yaxis_title='Amount ($)',
                barmode='group',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating budget vs actual chart: {str(e)}")
            return None
    
    def create_cash_flow_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create cash flow chart showing cumulative balance"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate cumulative balance
            df['cumulative_balance'] = df['amount'].cumsum()
            
            # Daily aggregation
            daily_data = df.groupby(df['date'].dt.date).agg({
                'amount': 'sum',
                'cumulative_balance': 'last'
            })
            
            fig = go.Figure()
            
            # Cumulative balance line
            fig.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['cumulative_balance'],
                mode='lines+markers',
                name='Cumulative Balance',
                line=dict(color=self.color_scheme['primary'], width=3),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title='Cash Flow Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative Balance ($)',
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cash flow chart: {str(e)}")
            return None
    
    def create_spending_heatmap(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create spending heatmap by day of week and hour"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            expenses['day_of_week'] = expenses['date'].dt.day_name()
            expenses['hour'] = expenses['date'].dt.hour
            
            # Create pivot table for heatmap
            heatmap_data = expenses.groupby(['day_of_week', 'hour'])['amount'].sum().unstack(fill_value=0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Spending ($)")
            ))
            
            fig.update_layout(
                title='Spending Patterns by Day and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating spending heatmap: {str(e)}")
            return None
    
    def create_forecast_chart(self, historical_data: List[Dict], predictions: Dict) -> Optional[go.Figure]:
        """Create forecast chart with historical and predicted data"""
        try:
            if not historical_data or not predictions:
                return None
                
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Historical monthly data
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            expenses['month'] = expenses['date'].dt.to_period('M')
            
            monthly_expenses = expenses.groupby('month')['amount'].sum()
            
            # Prepare forecast data
            forecast_data = predictions.get('monthly_forecast', {})
            last_month = monthly_expenses.index[-1] if not monthly_expenses.empty else pd.Period.now('M')
            
            # Generate future months
            future_months = [last_month + i for i in range(1, 7)]  # Next 6 months
            future_expenses = [forecast_data.get('expenses', 0)] * len(future_months)
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=[str(m) for m in monthly_expenses.index],
                y=monthly_expenses.values,
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=8)
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=[str(m) for m in future_months],
                y=future_expenses,
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_scheme['warning'], width=3, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Add confidence interval if available
            if 'confidence' in forecast_data:
                confidence = forecast_data['confidence']
                upper_bound = [exp * (1 + (1 - confidence)) for exp in future_expenses]
                lower_bound = [exp * confidence for exp in future_expenses]
                
                fig.add_trace(go.Scatter(
                    x=[str(m) for m in future_months],
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                title='Spending Forecast',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {str(e)}")
            return None
    
    def create_savings_goal_progress(self, transactions: List[Dict], savings_goal: float) -> Optional[go.Figure]:
        """Create savings goal progress chart"""
        try:
            if not transactions or savings_goal <= 0:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate monthly savings
            df['month'] = df['date'].dt.to_period('M')
            monthly_data = df.groupby('month')['amount'].sum()
            
            # Calculate cumulative savings
            cumulative_savings = monthly_data.cumsum()
            
            # Calculate goal line
            months_elapsed = len(monthly_data)
            goal_line = [savings_goal * (i + 1) for i in range(months_elapsed)]
            
            fig = go.Figure()
            
            # Actual savings
            fig.add_trace(go.Scatter(
                x=[str(m) for m in cumulative_savings.index],
                y=cumulative_savings.values,
                mode='lines+markers',
                name='Actual Savings',
                line=dict(color=self.color_scheme['success'], width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ))
            
            # Goal line
            fig.add_trace(go.Scatter(
                x=[str(m) for m in cumulative_savings.index],
                y=goal_line,
                mode='lines',
                name='Savings Goal',
                line=dict(color=self.color_scheme['warning'], width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Savings Goal Progress (${savings_goal:.0f}/month target)',
                xaxis_title='Month',
                yaxis_title='Cumulative Savings ($)',
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating savings goal progress chart: {str(e)}")
            return None
    
    def create_expense_distribution_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create expense amount distribution histogram"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            expenses = df[df['amount'] < 0]['amount'].abs()
            
            if expenses.empty:
                return None
            
            fig = go.Figure()
            
            # Histogram of expense amounts
            fig.add_trace(go.Histogram(
                x=expenses,
                nbinsx=30,
                name='Expense Distribution',
                marker_color=self.color_scheme['expenses'],
                opacity=0.7
            ))
            
            # Add median line
            median_expense = expenses.median()
            fig.add_vline(
                x=median_expense,
                line_dash="dash",
                line_color=self.color_scheme['primary'],
                annotation_text=f"Median: ${median_expense:.2f}"
            )
            
            # Add mean line
            mean_expense = expenses.mean()
            fig.add_vline(
                x=mean_expense,
                line_dash="dot",
                line_color=self.color_scheme['secondary'],
                annotation_text=f"Mean: ${mean_expense:.2f}"
            )
            
            fig.update_layout(
                title='Distribution of Expense Amounts',
                xaxis_title='Expense Amount ($)',
                yaxis_title='Frequency',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating expense distribution chart: {str(e)}")
            return None
    
    def create_merchant_analysis_chart(self, transactions: List[Dict], top_n: int = 10) -> Optional[go.Figure]:
        """Create top merchants spending analysis"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            expenses = df[df['amount'] < 0].copy()
            
            if expenses.empty or 'description' not in expenses.columns:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Group by merchant/description
            merchant_spending = expenses.groupby('description').agg({
                'amount': ['sum', 'count', 'mean']
            }).round(2)
            
            merchant_spending.columns = ['total_spent', 'transaction_count', 'avg_transaction']
            merchant_spending = merchant_spending.sort_values('total_spent', ascending=True).tail(top_n)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=merchant_spending['total_spent'],
                y=merchant_spending.index,
                orientation='h',
                name='Total Spent',
                marker_color=self.color_scheme['expenses'],
                text=[f"${x:.0f}" for x in merchant_spending['total_spent']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Merchants by Spending',
                xaxis_title='Total Amount Spent ($)',
                yaxis_title='Merchant',
                height=max(400, top_n * 40),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating merchant analysis chart: {str(e)}")
            return None
    
    def create_weekly_pattern_chart(self, transactions: List[Dict]) -> Optional[go.Figure]:
        """Create weekly spending pattern chart"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return None
                
            expenses['amount'] = expenses['amount'].abs()
            expenses['day_of_week'] = expenses['date'].dt.day_name()
            
            # Calculate daily averages
            daily_spending = expenses.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count'])
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_spending = daily_spending.reindex(day_order)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Total Spending by Day', 'Average Transaction by Day'),
                vertical_spacing=0.1
            )
            
            # Total spending
            fig.add_trace(
                go.Bar(
                    x=daily_spending.index,
                    y=daily_spending['sum'],
                    name='Total',
                    marker_color=self.color_scheme['expenses']
                ),
                row=1, col=1
            )
            
            # Average transaction
            fig.add_trace(
                go.Bar(
                    x=daily_spending.index,
                    y=daily_spending['mean'],
                    name='Average',
                    marker_color=self.color_scheme['secondary']
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Weekly Spending Patterns',
                height=600,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Day of Week", row=2, col=1)
            fig.update_yaxes(title_text="Total Amount ($)", row=1, col=1)
            fig.update_yaxes(title_text="Average Amount ($)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating weekly pattern chart: {str(e)}")
            return None
    
    def create_financial_dashboard(self, transactions: List[Dict], user_data: Dict) -> Optional[go.Figure]:
        """Create comprehensive financial dashboard"""
        try:
            if not transactions:
                return None
                
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Monthly Cash Flow',
                    'Category Breakdown',
                    'Daily Spending Trend',
                    'Savings Rate'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "indicator"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 1. Monthly Cash Flow
            expenses = df[df['amount'] < 0].copy()
            income = df[df['amount'] > 0].copy()
            
            if not expenses.empty:
                expenses['amount'] = expenses['amount'].abs()
                expenses['month'] = expenses['date'].dt.to_period('M')
                monthly_expenses = expenses.groupby('month')['amount'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=[str(m) for m in monthly_expenses.index],
                        y=monthly_expenses.values,
                        name='Expenses',
                        marker_color=self.color_scheme['expenses']
                    ),
                    row=1, col=1
                )
            
            if not income.empty:
                income['month'] = income['date'].dt.to_period('M')
                monthly_income = income.groupby('month')['amount'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=[str(m) for m in monthly_income.index],
                        y=monthly_income.values,
                        name='Income',
                        marker_color=self.color_scheme['income']
                    ),
                    row=1, col=1
                )
            
            # 2. Category Breakdown (Pie Chart)
            if not expenses.empty and 'category' in expenses.columns:
                category_totals = expenses.groupby('category')['amount'].sum()
                
                fig.add_trace(
                    go.Pie(
                        labels=category_totals.index,
                        values=category_totals.values,
                        name="Categories"
                    ),
                    row=1, col=2
                )
            
            # 3. Daily Spending Trend
            if not expenses.empty:
                daily_spending = expenses.groupby(expenses['date'].dt.date)['amount'].sum()
                
                fig.add_trace(
                    go.Scatter(
                        x=daily_spending.index,
                        y=daily_spending.values,
                        mode='lines',
                        name='Daily Spending',
                        line=dict(color=self.color_scheme['primary'])
                    ),
                    row=2, col=1
                )
            
            # 4. Savings Rate Indicator
            total_income = income['amount'].sum() if not income.empty else 0
            total_expenses = expenses['amount'].sum() if not expenses.empty else 0
            savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=savings_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Savings Rate (%)"},
                    delta={'reference': 20},  # 20% is a good savings rate
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': self.color_scheme['success']},
                        'steps': [
                            {'range': [0, 10], 'color': self.color_scheme['danger']},
                            {'range': [10, 20], 'color': self.color_scheme['warning']},
                            {'range': [20, 50], 'color': self.color_scheme['success']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 20
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Financial Dashboard Overview',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating financial dashboard: {str(e)}")
            return None
    
    def create_comparison_chart(self, current_data: List[Dict], previous_data: List[Dict]) -> Optional[go.Figure]:
        """Create comparison chart between two time periods"""
        try:
            if not current_data or not previous_data:
                return None
                
            # Process current period
            current_df = pd.DataFrame(current_data)
            current_expenses = current_df[current_df['amount'] < 0]['amount'].abs().sum()
            current_income = current_df[current_df['amount'] > 0]['amount'].sum()
            
            # Process previous period
            previous_df = pd.DataFrame(previous_data)
            previous_expenses = previous_df[previous_df['amount'] < 0]['amount'].abs().sum()
            previous_income = previous_df[previous_df['amount'] > 0]['amount'].sum()
            
            # Calculate changes
            expense_change = ((current_expenses - previous_expenses) / previous_expenses * 100) if previous_expenses > 0 else 0
            income_change = ((current_income - previous_income) / previous_income * 100) if previous_income > 0 else 0
            
            categories = ['Expenses', 'Income']
            current_values = [current_expenses, current_income]
            previous_values = [previous_expenses, previous_income]
            changes = [expense_change, income_change]
            
            fig = go.Figure()
            
            # Previous period bars
            fig.add_trace(go.Bar(
                x=categories,
                y=previous_values,
                name='Previous Period',
                marker_color=self.color_scheme['secondary'],
                opacity=0.7
            ))
            
            # Current period bars
            fig.add_trace(go.Bar(
                x=categories,
                y=current_values,
                name='Current Period',
                marker_color=self.color_scheme['primary'],
                opacity=0.7
            ))
            
            # Add change annotations
            for i, (cat, change) in enumerate(zip(categories, changes)):
                fig.add_annotation(
                    x=cat,
                    y=max(current_values[i], previous_values[i]) * 1.1,
                    text=f"{change:+.1f}%",
                    showarrow=False,
                    font=dict(color='green' if change > 0 and cat == 'Income' else 'red' if change > 0 else 'green')
                )
            
            fig.update_layout(
                title='Period Comparison',
                xaxis_title='Category',
                yaxis_title='Amount ($)',
                barmode='group',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
            return None
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[str(m) for m in future_months
