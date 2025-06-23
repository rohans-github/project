"""
Predictive Modeling Engine
Forecasts future spending, income, and savings using machine learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictiveModel:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_predictions(self, transactions: List[Dict], user_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive financial predictions
        """
        if not transactions:
            return {'error': 'No transaction data available for predictions'}
            
        try:
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            predictions = {
                'monthly_forecast': self._forecast_monthly_spending(df, user_data),
                'goal_prediction': self._predict_goal_achievement(df, user_data),
                'category_forecasts': self._forecast_by_category(df),
                'cash_flow_prediction': self._predict_cash_flow(df),
                'trend_analysis': self._analyze_future_trends(df),
                'risk_assessment': self._assess_financial_risks(df, user_data)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {'error': str(e)}
    
    def _forecast_monthly_spending(self, df: pd.DataFrame, user_data: Dict) -> Dict[str, float]:
        """Forecast next month's expenses and savings"""
        try:
            # Prepare monthly aggregated data
            df['month'] = df['date'].dt.to_period('M')
            monthly_data = df.groupby('month').agg({
                'amount': ['sum', 'count']
            }).reset_index()
            
            monthly_data.columns = ['month', 'total_amount', 'transaction_count']
            monthly_data['expenses'] = monthly_data['total_amount'].apply(lambda x: abs(x) if x < 0 else 0)
            monthly_data['income'] = monthly_data['total_amount'].apply(lambda x: x if x > 0 else 0)
            
            if len(monthly_data) < 2:
                # Use simple averages if insufficient data
                expenses = df[df['amount'] < 0]['amount'].abs().sum()
                income = df[df['amount'] > 0]['amount'].sum()
                days_of_data = (df['date'].max() - df['date'].min()).days
                
                if days_of_data > 0:
                    monthly_expenses = (expenses / days_of_data) * 30
                    monthly_income = (income / days_of_data) * 30
                else:
                    monthly_expenses = expenses
                    monthly_income = income
                    
                return {
                    'expenses': monthly_expenses,
                    'income': monthly_income,
                    'savings': monthly_income - monthly_expenses,
                    'confidence': 0.5
                }
            
            # Use trend-based forecasting
            months_numeric = np.arange(len(monthly_data))
            
            # Forecast expenses
            expense_model = LinearRegression()
            expense_model.fit(months_numeric.reshape(-1, 1), monthly_data['expenses'])
            next_month_expenses = expense_model.predict([[len(monthly_data)]])[0]
            
            # Forecast income
            income_model = LinearRegression()
            income_model.fit(months_numeric.reshape(-1, 1), monthly_data['income'])
            next_month_income = income_model.predict([[len(monthly_data)]])[0]
            
            # Calculate confidence based on model performance
            expense_predictions = expense_model.predict(months_numeric.reshape(-1, 1))
            expense_mae = mean_absolute_error(monthly_data['expenses'], expense_predictions)
            confidence = max(0.1, 1 - (expense_mae / monthly_data['expenses'].mean()))
            
            return {
                'expenses': max(0, next_month_expenses),
                'income': max(0, next_month_income),
                'savings': next_month_income - next_month_expenses,
                'confidence': min(1.0, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting monthly spending: {str(e)}")
            return {'expenses': 0, 'income': 0, 'savings': 0, 'confidence': 0}
    
    def _predict_goal_achievement(self, df: pd.DataFrame, user_data: Dict) -> Dict[str, Any]:
        """Predict likelihood of achieving financial goals"""
        try:
            savings_goal = user_data.get('savings_goal', 0)
            monthly_income = user_data.get('monthly_income', 0)
            
            if savings_goal <= 0:
                return {'probability': 0, 'message': 'No savings goal set'}
            
            # Calculate current savings rate
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            if income <= 0:
                return {'probability': 0, 'message': 'No income data available'}
            
            current_savings = income - expenses
            days_of_data = (df['date'].max() - df['date'].min()).days
            
            if days_of_data > 0:
                monthly_savings = (current_savings / days_of_data) * 30
            else:
                monthly_savings = current_savings
            
            # Calculate probability based on current trajectory
            if monthly_savings >= savings_goal:
                probability = 0.9  # High probability if already meeting goal
            elif monthly_savings >= savings_goal * 0.8:
                probability = 0.7  # Good chance if close to goal
            elif monthly_savings >= savings_goal * 0.5:
                probability = 0.5  # Moderate chance
            elif monthly_savings >= 0:
                probability = 0.3  # Low chance but positive savings
            else:
                probability = 0.1  # Very low chance if spending exceeds income
            
            # Adjust based on trend
            if len(df) > 30:  # If we have enough data
                recent_data = df[df['date'] >= df['date'].max() - timedelta(days=30)]
                older_data = df[df['date'] < df['date'].max() - timedelta(days=30)]
                
                if not recent_data.empty and not older_data.empty:
                    recent_savings_rate = (recent_data[recent_data['amount'] > 0]['amount'].sum() - 
                                         recent_data[recent_data['amount'] < 0]['amount'].abs().sum())
                    older_savings_rate = (older_data[older_data['amount'] > 0]['amount'].sum() - 
                                        older_data[older_data['amount'] < 0]['amount'].abs().sum())
                    
                    if recent_savings_rate > older_savings_rate:
                        probability = min(1.0, probability * 1.2)  # Improving trend
                    elif recent_savings_rate < older_savings_rate:
                        probability = max(0.1, probability * 0.8)  # Declining trend
            
            return {
                'probability': probability,
                'current_monthly_savings': monthly_savings,
                'goal_gap': savings_goal - monthly_savings,
                'message': self._generate_goal_message(probability, monthly_savings, savings_goal)
            }
            
        except Exception as e:
            logger.error(f"Error predicting goal achievement: {str(e)}")
            return {'probability': 0, 'message': 'Error calculating goal prediction'}
    
    def _forecast_by_category(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Forecast spending by category"""
        category_forecasts = {}
        
        try:
            if 'category' not in df.columns:
                return category_forecasts
            
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            expenses['month'] = expenses['date'].dt.to_period('M')
            
            for category in expenses['category'].unique():
                cat_data = expenses[expenses['category'] == category]
                monthly_spending = cat_data.groupby('month')['amount'].sum()
                
                if len(monthly_spending) >= 2:
                    # Simple trend-based forecast
                    months = np.arange(len(monthly_spending))
                    model = LinearRegression()
                    model.fit(months.reshape(-1, 1), monthly_spending.values)
                    
                    next_month_forecast = model.predict([[len(monthly_spending)]])[0]
                    
                    # Calculate trend
                    slope = model.coef_[0]
                    trend = 'increasing' if slope > 5 else 'decreasing' if slope < -5 else 'stable'
                    
                    category_forecasts[category] = {
                        'forecast': max(0, next_month_forecast),
                        'trend': trend,
                        'current_avg': monthly_spending.mean(),
                        'confidence': min(1.0, max(0.3, 1 - abs(slope) / monthly_spending.mean()))
                    }
                else:
                    # Use simple average if insufficient data
                    avg_spending = cat_data['amount'].mean()
                    category_forecasts[category] = {
                        'forecast': avg_spending,
                        'trend': 'unknown',
                        'current_avg': avg_spending,
                        'confidence': 0.5
                    }
                    
        except Exception as e:
            logger.error(f"Error forecasting by category: {str(e)}")
            
        return category_forecasts
    
    def _predict_cash_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict future cash flow patterns"""
        try:
            df['week'] = df['date'].dt.isocalendar().week
            weekly_flow = df.groupby('week')['amount'].sum()
            
            if len(weekly_flow) < 3:
                return {'error': 'Insufficient data for cash flow prediction'}
            
            # Predict next few weeks
            weeks = np.arange(len(weekly_flow))
            model = LinearRegression()
            model.fit(weeks.reshape(-1, 1), weekly_flow.values)
            
            # Forecast next 4 weeks
            future_weeks = np.arange(len(weekly_flow), len(weekly_flow) + 4)
            future_cash_flow = model.predict(future_weeks.reshape(-1, 1))
            
            # Calculate cumulative cash flow
            current_balance = weekly_flow.sum()
            cumulative_forecast = [current_balance]
            
            for flow in future_cash_flow:
                cumulative_forecast.append(cumulative_forecast[-1] + flow)
            
            return {
                'weekly_forecasts': future_cash_flow.tolist(),
                'cumulative_balance': cumulative_forecast[1:],
                'trend': 'positive' if model.coef_[0] > 0 else 'negative',
                'volatility': weekly_flow.std()
            }
            
        except Exception as e:
            logger.error(f"Error predicting cash flow: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_future_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze future spending trends"""
        try:
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            expenses['month'] = expenses['date'].dt.to_period('M')
            
            monthly_totals = expenses.groupby('month')['amount'].sum()
            
            if len(monthly_totals) < 3:
                return {'message': 'Insufficient data for trend analysis'}
            
            # Calculate trend metrics
            months = np.arange(len(monthly_totals))
            model = LinearRegression()
            model.fit(months.reshape(-1, 1), monthly_totals.values)
            
            slope = model.coef_[0]
            r_squared = model.score(months.reshape(-1, 1), monthly_totals.values)
            
            # Seasonal analysis
            if len(monthly_totals) >= 12:
                seasonal_pattern = self._detect_seasonal_patterns(monthly_totals)
            else:
                seasonal_pattern = None
            
            return {
                'monthly_trend': slope,
                'trend_strength': r_squared,
                'trend_direction': 'increasing' if slope > 10 else 'decreasing' if slope < -10 else 'stable',
                'seasonal_pattern': seasonal_pattern,
                'volatility': monthly_totals.std(),
                'prediction_confidence': r_squared
            }
            
        except Exception as e:
            logger.error(f"Error analyzing future trends: {str(e)}")
            return {'error': str(e)}
    
    def _assess_financial_risks(self, df: pd.DataFrame, user_data: Dict) -> Dict[str, Any]:
        """Assess financial risks based on spending patterns"""
        risks = []
        risk_score = 0
        
        try:
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            # Risk 1: High expense ratio
            if income > 0:
                expense_ratio = expenses / income
                if expense_ratio > 0.9:
                    risks.append("High expense ratio - spending >90% of income")
                    risk_score += 30
                elif expense_ratio > 0.8:
                    risks.append("Moderate expense ratio - spending >80% of income")
                    risk_score += 15
            
            # Risk 2: High spending volatility
            expenses_df = df[df['amount'] < 0].copy()
            if not expenses_df.empty:
                expenses_df['amount'] = expenses_df['amount'].abs()
                expense_std = expenses_df['amount'].std()
                expense_mean = expenses_df['amount'].mean()
                
                if expense_mean > 0 and expense_std / expense_mean > 1.5:
                    risks.append("High spending volatility detected")
                    risk_score += 20
            
            # Risk 3: Increasing spending trend
            if len(df) > 30:
                recent_expenses = df[df['date'] >= df['date'].max() - timedelta(days=30)]
                older_expenses = df[df['date'] < df['date'].max() - timedelta(days=30)]
                
                if not recent_expenses.empty and not older_expenses.empty:
                    recent_avg = recent_expenses[recent_expenses['amount'] < 0]['amount'].abs().mean()
                    older_avg = older_expenses[older_expenses['amount'] < 0]['amount'].abs().mean()
                    
                    if recent_avg > older_avg * 1.2:
                        risks.append("Spending trend increasing rapidly")
                        risk_score += 25
            
            # Risk 4: Lack of emergency fund indicator
            savings_goal = user_data.get('savings_goal', 0)
            monthly_income = user_data.get('monthly_income', 0)
            
            if monthly_income > 0 and savings_goal < monthly_income * 0.1:
                risks.append("Low savings goal relative to income")
                risk_score += 15
            
            # Determine overall risk level
            if risk_score >= 60:
                risk_level = 'High'
            elif risk_score >= 30:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'risk_level': risk_level,
                'risk_score': min(100, risk_score),
                'identified_risks': risks,
                'recommendations': self._generate_risk_recommendations(risks, risk_level)
            }
            
        except Exception as e:
            logger.error(f"Error assessing financial risks: {str(e)}")
            return {'risk_level': 'Unknown', 'error': str(e)}
    
    def _detect_seasonal_patterns(self, monthly_data: pd.Series) -> Dict[str, Any]:
        """Detect seasonal spending patterns"""
        try:
            # Convert period index to month numbers
            months = [period.month for period in monthly_data.index]
            values = monthly_data.values
            
            # Group by month and calculate averages
            month_averages = {}
            for month, value in zip(months, values):
                if month not in month_averages:
                    month_averages[month] = []
                month_averages[month].append(value)
            
            # Calculate average spending by month
            seasonal_averages = {month: np.mean(values) for month, values in month_averages.items()}
            
            # Find highest and lowest spending months
            highest_month = max(seasonal_averages, key=seasonal_averages.get)
            lowest_month = min(seasonal_averages, key=seasonal_averages.get)
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            return {
                'highest_spending_month': month_names[highest_month - 1],
                'lowest_spending_month': month_names[lowest_month - 1],
                'seasonal_variation': (seasonal_averages[highest_month] - 
                                     seasonal_averages[lowest_month]) / seasonal_averages[lowest_month],
                'monthly_averages': {month_names[month - 1]: avg 
                                   for month, avg in seasonal_averages.items()}
            }
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {str(e)}")
            return None
    
    def _generate_goal_message(self, probability: float, current_savings: float, goal: float) -> str:
        """Generate message about goal achievement probability"""
        if probability >= 0.8:
            return f"Excellent! You're on track to meet your savings goal of ${goal:.2f}"
        elif probability >= 0.6:
            return f"Good progress! You're likely to achieve your savings goal with current habits"
        elif probability >= 0.4:
            return f"Moderate chance of reaching your ${goal:.2f} goal. Consider reducing expenses"
        elif probability >= 0.2:
            return f"Low probability of meeting savings goal. Significant changes needed"
        else:
            return f"Current spending pattern makes it very difficult to reach your savings goal"
    
    def _generate_risk_recommendations(self, risks: List[str], risk_level: str) -> List[str]:
        """Generate recommendations based on identified risks"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("Create an emergency budget plan immediately")
            recommendations.append("Consider seeking financial counseling")
            recommendations.append("Review all non-essential expenses for immediate cuts")
        
        elif risk_level == 'Medium':
            recommendations.append("Build an emergency fund of 3-6 months expenses")
            recommendations.append("Review and optimize your spending categories")
            recommendations.append("Consider additional income sources")
        
        else:  # Low risk
            recommendations.append("Continue monitoring your spending patterns")
            recommendations.append("Consider increasing your savings rate")
            recommendations.append("Explore investment opportunities")
        
        # Specific recommendations based on identified risks
        for risk in risks:
            if "volatility" in risk.lower():
                recommendations.append("Try to stabilize your spending patterns")
            elif "increasing" in risk.lower():
                recommendations.append("Identify the cause of increased spending")
            elif "savings goal" in risk.lower():
                recommendations.append("Increase your savings target to at least 10% of income")
        
        return recommendations[:5]  # Limit to top 5 recommendations
