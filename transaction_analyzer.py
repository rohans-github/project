"""
Transaction Analysis Engine
Analyzes spending patterns, detects anomalies, and identifies trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TransactionAnalyzer:
    def __init__(self):
        self.categories = [
            'Food', 'Transportation', 'Entertainment', 'Utilities', 
            'Healthcare', 'Shopping', 'Travel', 'Education', 'Other'
        ]
        
    def analyze_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive transaction analysis
        """
        if not transactions:
            return {}
            
        try:
            df = pd.DataFrame(transactions)
            
            # Ensure required columns exist
            if 'amount' not in df.columns or 'date' not in df.columns:
                return {'error': 'Missing required columns (amount, date)'}
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            df['day_of_week'] = df['date'].dt.day_name()
            
            analysis_results = {
                'patterns': self._identify_spending_patterns(df),
                'anomalies': self._detect_anomalies(df),
                'category_analysis': self._analyze_categories(df),
                'trends': self._analyze_trends(df),
                'insights': self._generate_insights(df),
                'statistics': self._calculate_statistics(df)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in transaction analysis: {str(e)}")
            return {'error': str(e)}
    
    def _identify_spending_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify spending patterns in the data"""
        patterns = []
        
        try:
            # Expense data only
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return patterns
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Day of week patterns
            daily_spending = expenses.groupby('day_of_week')['amount'].mean()
            highest_day = daily_spending.idxmax()
            lowest_day = daily_spending.idxmin()
            
            patterns.append(f"Highest spending day: {highest_day} (${daily_spending[highest_day]:.2f} avg)")
            patterns.append(f"Lowest spending day: {lowest_day} (${daily_spending[lowest_day]:.2f} avg)")
            
            # Monthly patterns
            monthly_spending = expenses.groupby('month')['amount'].sum()
            if len(monthly_spending) > 1:
                trend = "increasing" if monthly_spending.iloc[-1] > monthly_spending.iloc[0] else "decreasing"
                patterns.append(f"Monthly spending trend: {trend}")
            
            # Frequency patterns
            transaction_freq = len(expenses) / max(1, (expenses['date'].max() - expenses['date'].min()).days)
            patterns.append(f"Average transaction frequency: {transaction_freq:.1f} transactions per day")
            
            # Large transaction patterns
            median_amount = expenses['amount'].median()
            large_transactions = expenses[expenses['amount'] > median_amount * 3]
            if not large_transactions.empty:
                patterns.append(f"Large transactions (>3x median): {len(large_transactions)} found")
                
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            patterns.append("Error analyzing patterns")
            
        return patterns
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[str]:
        """Detect anomalous transactions"""
        anomalies = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return anomalies
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Statistical anomalies (Z-score method)
            mean_amount = expenses['amount'].mean()
            std_amount = expenses['amount'].std()
            
            if std_amount > 0:
                z_scores = np.abs((expenses['amount'] - mean_amount) / std_amount)
                statistical_anomalies = expenses[z_scores > 2.5]
                
                for _, row in statistical_anomalies.iterrows():
                    anomalies.append(
                        f"Unusual amount: ${row['amount']:.2f} on {row['date'].strftime('%Y-%m-%d')}"
                    )
            
            # Frequency anomalies
            daily_counts = expenses.groupby(expenses['date'].dt.date).size()
            avg_daily_transactions = daily_counts.mean()
            
            high_frequency_days = daily_counts[daily_counts > avg_daily_transactions * 2]
            for date, count in high_frequency_days.items():
                anomalies.append(f"High transaction day: {count} transactions on {date}")
            
            # Category anomalies
            if 'category' in expenses.columns:
                category_spending = expenses.groupby('category')['amount'].sum()
                total_spending = category_spending.sum()
                
                for category, amount in category_spending.items():
                    percentage = (amount / total_spending) * 100
                    if percentage > 50:
                        anomalies.append(f"High category concentration: {category} ({percentage:.1f}% of spending)")
                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            anomalies.append("Error detecting anomalies")
            
        return anomalies[:5]  # Limit to top 5 anomalies
    
    def _analyze_categories(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze spending by category"""
        category_analysis = {}
        
        try:
            if 'category' not in df.columns:
                # Categorize based on description if category not available
                df['category'] = df.get('description', '').apply(self._categorize_transaction)
            
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty:
                return category_analysis
                
            expenses['amount'] = expenses['amount'].abs()
            
            category_stats = expenses.groupby('category').agg({
                'amount': ['sum', 'mean', 'count', 'std']
            }).round(2)
            
            for category in category_stats.index:
                category_analysis[category] = {
                    'total': category_stats.loc[category, ('amount', 'sum')],
                    'average': category_stats.loc[category, ('amount', 'mean')],
                    'count': category_stats.loc[category, ('amount', 'count')],
                    'std': category_stats.loc[category, ('amount', 'std')]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing categories: {str(e)}")
            
        return category_analysis
    
    def _analyze_trends(self, df: pd.DataFrame) -> List[str]:
        """Analyze spending trends over time"""
        trends = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            if expenses.empty or len(expenses) < 2:
                return trends
                
            expenses['amount'] = expenses['amount'].abs()
            
            # Monthly trends
            monthly_spending = expenses.groupby('month')['amount'].sum()
            if len(monthly_spending) >= 2:
                recent_month = monthly_spending.iloc[-1]
                previous_month = monthly_spending.iloc[-2]
                change = ((recent_month - previous_month) / previous_month) * 100
                
                if abs(change) > 5:
                    direction = "increased" if change > 0 else "decreased"
                    trends.append(f"Monthly spending {direction} by {abs(change):.1f}%")
            
            # Weekly trends
            expenses['week'] = expenses['date'].dt.isocalendar().week
            weekly_spending = expenses.groupby('week')['amount'].sum()
            
            if len(weekly_spending) >= 4:
                # Calculate trend using linear regression slope
                weeks = np.arange(len(weekly_spending))
                spending_values = weekly_spending.values
                slope, _ = np.polyfit(weeks, spending_values, 1)
                
                if abs(slope) > 10:
                    direction = "increasing" if slope > 0 else "decreasing"
                    trends.append(f"Weekly spending shows {direction} trend")
            
            # Category trends
            if 'category' in expenses.columns and len(monthly_spending) >= 2:
                category_trends = {}
                for category in expenses['category'].unique():
                    cat_monthly = expenses[expenses['category'] == category].groupby('month')['amount'].sum()
                    if len(cat_monthly) >= 2:
                        cat_change = ((cat_monthly.iloc[-1] - cat_monthly.iloc[-2]) / 
                                    max(cat_monthly.iloc[-2], 1)) * 100
                        if abs(cat_change) > 20:
                            category_trends[category] = cat_change
                
                if category_trends:
                    top_increasing = max(category_trends.items(), key=lambda x: x[1])
                    trends.append(f"Fastest growing category: {top_increasing[0]} (+{top_increasing[1]:.1f}%)")
                    
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            trends.append("Error analyzing trends")
            
        return trends
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from the data"""
        insights = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            income = df[df['amount'] > 0].copy()
            
            if not expenses.empty:
                expenses['amount'] = expenses['amount'].abs()
                
                # Spending insights
                total_expenses = expenses['amount'].sum()
                avg_transaction = expenses['amount'].mean()
                
                insights.append(f"Average transaction size: ${avg_transaction:.2f}")
                
                # High-value transaction insight
                high_value_threshold = avg_transaction * 2
                high_value_transactions = expenses[expenses['amount'] > high_value_threshold]
                high_value_total = high_value_transactions['amount'].sum()
                high_value_percentage = (high_value_total / total_expenses) * 100
                
                if high_value_percentage > 20:
                    insights.append(
                        f"{len(high_value_transactions)} large transactions account for "
                        f"{high_value_percentage:.1f}% of total spending"
                    )
                
                # Frequency insights
                date_range = (expenses['date'].max() - expenses['date'].min()).days
                if date_range > 0:
                    transaction_frequency = len(expenses) / date_range
                    insights.append(f"Transaction frequency: {transaction_frequency:.1f} transactions per day")
            
            # Income vs expenses
            if not income.empty and not expenses.empty:
                total_income = income['amount'].sum()
                total_expenses = expenses['amount'].sum()
                savings_rate = ((total_income - total_expenses) / total_income) * 100
                
                if savings_rate > 0:
                    insights.append(f"Current savings rate: {savings_rate:.1f}%")
                else:
                    insights.append("Warning: Expenses exceed income")
                    
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            
        return insights
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key financial statistics"""
        stats = {}
        
        try:
            expenses = df[df['amount'] < 0].copy()
            income = df[df['amount'] > 0].copy()
            
            if not expenses.empty:
                expenses['amount'] = expenses['amount'].abs()
                stats.update({
                    'total_expenses': expenses['amount'].sum(),
                    'avg_expense': expenses['amount'].mean(),
                    'median_expense': expenses['amount'].median(),
                    'expense_std': expenses['amount'].std(),
                    'transaction_count': len(expenses)
                })
            
            if not income.empty:
                stats.update({
                    'total_income': income['amount'].sum(),
                    'avg_income': income['amount'].mean(),
                    'income_transactions': len(income)
                })
            
            # Calculate additional metrics
            if 'total_income' in stats and 'total_expenses' in stats:
                stats['net_savings'] = stats['total_income'] - stats['total_expenses']
                stats['savings_rate'] = (stats['net_savings'] / stats['total_income']) * 100
                
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            
        return stats
    
    def _categorize_transaction(self, description: str) -> str:
        """Categorize transaction based on description"""
        if not description:
            return 'Other'
            
        description_lower = description.lower()
        
        # Food-related keywords
        food_keywords = ['restaurant', 'grocery', 'food', 'cafe', 'pizza', 'mcdonalds', 'starbucks']
        if any(keyword in description_lower for keyword in food_keywords):
            return 'Food'
        
        # Transportation keywords
        transport_keywords = ['gas', 'fuel', 'uber', 'taxi', 'bus', 'train', 'parking']
        if any(keyword in description_lower for keyword in transport_keywords):
            return 'Transportation'
        
        # Entertainment keywords
        entertainment_keywords = ['movie', 'netflix', 'spotify', 'game', 'entertainment']
        if any(keyword in description_lower for keyword in entertainment_keywords):
            return 'Entertainment'
        
        # Utilities keywords
        utility_keywords = ['electric', 'water', 'internet', 'phone', 'utility']
        if any(keyword in description_lower for keyword in utility_keywords):
            return 'Utilities'
        
        # Healthcare keywords
        health_keywords = ['doctor', 'pharmacy', 'hospital', 'medical', 'health']
        if any(keyword in description_lower for keyword in health_keywords):
            return 'Healthcare'
        
        return 'Other'
    
    def get_category_recommendations(self, transactions: List[Dict]) -> Dict[str, List[str]]:
        """Get recommendations for each spending category"""
        recommendations = defaultdict(list)
        
        try:
            analysis = self.analyze_categories(transactions)
            
            for category, data in analysis.items():
                total = data.get('total', 0)
                count = data.get('count', 0)
                
                if category == 'Food' and total > 800:  # High food spending
                    recommendations[category].append("Consider meal planning to reduce food costs")
                    recommendations[category].append("Look for grocery discounts and bulk buying opportunities")
                
                elif category == 'Transportation' and total > 500:
                    recommendations[category].append("Consider carpooling or public transport options")
                    recommendations[category].append("Review fuel efficiency and maintenance costs")
                
                elif category == 'Entertainment' and total > 300:
                    recommendations[category].append("Look for free or low-cost entertainment alternatives")
                    recommendations[category].append("Review subscription services for unused accounts")
                
                elif category == 'Shopping' and count > 20:
                    recommendations[category].append("Consider a waiting period before purchases")
                    recommendations[category].append("Compare prices across different retailers")
                    
        except Exception as e:
            logger.error(f"Error generating category recommendations: {str(e)}")
            
        return dict(recommendations)
