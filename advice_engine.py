"""
AI-Powered Financial Advice Engine
Generates personalized financial advice based on transaction analysis and user goals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdviceEngine:
    def __init__(self):
        self.advice_rules = {
            'savings': self._generate_savings_advice,
            'spending': self._generate_spending_advice,
            'budgeting': self._generate_budgeting_advice,
            'investment': self._generate_investment_advice,
            'debt': self._generate_debt_advice
        }
        
        # Financial best practices thresholds
        self.thresholds = {
            'savings_rate_excellent': 0.20,
            'savings_rate_good': 0.15,
            'savings_rate_minimum': 0.10,
            'housing_ratio_max': 0.30,
            'debt_ratio_max': 0.36,
            'emergency_fund_months': 6,
            'high_expense_threshold': 2.0  # 2x median
        }
    
    def generate_advice(self, transactions: List[Dict], user_data: Dict, 
                       analysis_results: Dict = None) -> Dict[str, List[str]]:
        """
        Generate comprehensive personalized financial advice
        """
        if not transactions:
            return {'error': ['No transaction data available for advice generation']}
        
        try:
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            # Generate different types of advice
            advice = {
                'priority_recommendations': self._get_priority_recommendations(df, user_data, analysis_results),
                'savings_opportunities': self._identify_savings_opportunities(df, user_data),
                'budgeting_tips': self._generate_budgeting_tips(df, user_data),
                'investment_suggestions': self._generate_investment_suggestions(df, user_data),
                'warnings': self._generate_warnings(df, user_data),
                'quick_wins': self._identify_quick_wins(df, user_data),
                'long_term_strategies': self._suggest_long_term_strategies(df, user_data)
            }
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating advice: {str(e)}")
            return {'error': [str(e)]}
    
    def _get_priority_recommendations(self, df: pd.DataFrame, user_data: Dict, 
                                    analysis_results: Dict = None) -> List[str]:
        """Generate top priority recommendations"""
        recommendations = []
        
        try:
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            # Priority 1: Emergency fund
            if income > 0:
                savings_rate = (income - expenses) / income
                monthly_income = user_data.get('monthly_income', income)
                
                if savings_rate < self.thresholds['savings_rate_minimum']:
                    recommendations.append(
                        f"URGENT: Increase savings rate to at least 10%. "
                        f"Current rate: {savings_rate:.1%}"
                    )
                
                # Emergency fund recommendation
                estimated_monthly_expenses = expenses * (30 / max(1, (df['date'].max() - df['date'].min()).days))
                emergency_fund_needed = estimated_monthly_expenses * self.thresholds['emergency_fund_months']
                
                recommendations.append(
                    f"Build emergency fund of ${emergency_fund_needed:,.2f} "
                    f"({self.thresholds['emergency_fund_months']} months of expenses)"
                )
            
            # Priority 2: High-impact spending reductions
            high_impact_savings = self._identify_high_impact_savings(df)
            if high_impact_savings:
                recommendations.extend(high_impact_savings[:2])
            
            # Priority 3: Debt management (if detected)
            debt_advice = self._analyze_debt_patterns(df)
            if debt_advice:
                recommendations.extend(debt_advice[:2])
            
            # Priority 4: Goal alignment
            savings_goal = user_data.get('savings_goal', 0)
            if savings_goal > 0 and income > 0:
                current_monthly_savings = (income - expenses) * (30 / max(1, (df['date'].max() - df['date'].min()).days))
                if current_monthly_savings < savings_goal:
                    gap = savings_goal - current_monthly_savings
                    recommendations.append(
                        f"Increase monthly savings by ${gap:.2f} to reach your goal of ${savings_goal:.2f}"
                    )
                    
        except Exception as e:
            logger.error(f"Error generating priority recommendations: {str(e)}")
            recommendations.append("Error generating priority recommendations")
            
        return recommendations[:5]  # Top 5 priorities
    
    def _identify_savings_opportunities(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Identify specific opportunities to save money"""
        opportunities = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            
            # Category-based opportunities
            if 'category' in expenses.columns:
                category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
                
                for category, total in category_totals.head(5).items():
                    category_opportunities = self._get_category_savings_tips(category, total, expenses)
                    opportunities.extend(category_opportunities)
            
            # Frequency-based opportunities
            frequent_merchants = expenses['description'].value_counts().head(5) if 'description' in expenses.columns else pd.Series()
            for merchant, count in frequent_merchants.items():
                if count >= 10:  # Frequent transactions
                    merchant_total = expenses[expenses['description'] == merchant]['amount'].sum()
                    opportunities.append(
                        f"Review {merchant} spending: {count} transactions totaling ${merchant_total:.2f}"
                    )
            
            # Timing-based opportunities
            weekend_spending = expenses[expenses['date'].dt.weekday >= 5]['amount'].sum()
            total_spending = expenses['amount'].sum()
            
            if weekend_spending > total_spending * 0.4:
                opportunities.append(
                    f"Weekend spending is high (${weekend_spending:.2f}). "
                    f"Consider planning weekend activities in advance"
                )
            
            # Subscription detection
            subscription_opportunities = self._detect_subscriptions(expenses)
            opportunities.extend(subscription_opportunities)
            
        except Exception as e:
            logger.error(f"Error identifying savings opportunities: {str(e)}")
            
        return opportunities[:8]  # Top 8 opportunities
    
    def _generate_budgeting_tips(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Generate budgeting tips based on spending patterns"""
        tips = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            income = df[df['amount'] > 0]['amount'].sum()
            
            # 50/30/20 rule analysis
            if income > 0:
                needs_budget = income * 0.5
                wants_budget = income * 0.3
                savings_budget = income * 0.2
                
                current_savings = income - expenses['amount'].sum()
                
                tips.append(
                    f"Try the 50/30/20 rule: ${needs_budget:.2f} needs, "
                    f"${wants_budget:.2f} wants, ${savings_budget:.2f} savings"
                )
                
                if current_savings < savings_budget:
                    shortfall = savings_budget - current_savings
                    tips.append(f"Increase savings by ${shortfall:.2f} to follow 50/30/20 rule")
            
            # Category-specific budgeting
            if 'category' in expenses.columns:
                category_spending = expenses.groupby('category')['amount'].sum()
                
                # Food budgeting
                if 'Food' in category_spending:
                    food_spending = category_spending['Food']
                    recommended_food_budget = income * 0.10 if income > 0 else food_spending * 0.8
                    
                    if food_spending > recommended_food_budget:
                        tips.append(
                            f"Food spending (${food_spending:.2f}) exceeds recommended "
                            f"10% of income (${recommended_food_budget:.2f}). Try meal planning"
                        )
                
                # Transportation budgeting
                if 'Transportation' in category_spending:
                    transport_spending = category_spending['Transportation']
                    recommended_transport_budget = income * 0.15 if income > 0 else transport_spending * 0.8
                    
                    if transport_spending > recommended_transport_budget:
                        tips.append(
                            f"Transportation costs (${transport_spending:.2f}) are high. "
                            f"Consider carpooling or public transport"
                        )
            
            # Cash flow budgeting
            tips.extend(self._generate_cash_flow_tips(df))
            
            # Envelope budgeting suggestion
            if len(expenses) > 50:  # If enough transactions
                tips.append(
                    "Consider envelope budgeting: allocate specific amounts to spending categories"
                )
            
        except Exception as e:
            logger.error(f"Error generating budgeting tips: {str(e)}")
            
        return tips[:6]
    
    def _generate_investment_suggestions(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Generate investment suggestions based on financial situation"""
        suggestions = []
        
        try:
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            if income <= 0:
                return suggestions
            
            savings_rate = (income - expenses) / income
            monthly_income = user_data.get('monthly_income', income)
            
            # Basic investment readiness
            if savings_rate >= self.thresholds['savings_rate_good']:
                suggestions.append(
                    "Great savings rate! Consider investing excess funds in index funds or ETFs"
                )
                
                # Emergency fund check
                monthly_expenses = expenses * (30 / max(1, (df['date'].max() - df['date'].min()).days))
                emergency_fund_target = monthly_expenses * 3
                
                current_savings = income - expenses
                if current_savings > emergency_fund_target:
                    suggestions.append(
                        "You have good emergency fund coverage. "
                        "Consider investing in diversified portfolio"
                    )
                
                # Age-based suggestions (if age available)
                if monthly_income > 5000:
                    suggestions.append(
                        "With higher income, consider maxing out retirement accounts (401k, IRA)"
                    )
                    suggestions.append(
                        "Look into tax-advantaged accounts and employer matching programs"
                    )
                
            elif savings_rate >= self.thresholds['savings_rate_minimum']:
                suggestions.append(
                    "Start with low-cost index funds once you have 3-month emergency fund"
                )
                suggestions.append(
                    "Consider automatic investing to build discipline"
                )
            else:
                suggestions.append(
                    "Focus on increasing savings rate before investing. "
                    "Pay off high-interest debt first"
                )
            
            # Risk tolerance suggestions
            expense_volatility = df[df['amount'] < 0]['amount'].std()
            if expense_volatility < df[df['amount'] < 0]['amount'].abs().mean() * 0.5:
                suggestions.append(
                    "Your spending is stable - you may be able to take on moderate investment risk"
                )
            
        except Exception as e:
            logger.error(f"Error generating investment suggestions: {str(e)}")
            
        return suggestions[:4]
    
    def _generate_warnings(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Generate financial warnings based on concerning patterns"""
        warnings = []
        
        try:
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            # Warning 1: Spending exceeds income
            if expenses > income:
                overspend = expenses - income
                warnings.append(
                    f"WARNING: Spending exceeds income by ${overspend:.2f}. "
                    f"Immediate budget review needed"
                )
            
            # Warning 2: Very low savings rate
            if income > 0:
                savings_rate = (income - expenses) / income
                if savings_rate < 0.05:  # Less than 5%
                    warnings.append(
                        f"CRITICAL: Savings rate is only {savings_rate:.1%}. "
                        f"Financial vulnerability is high"
                    )
            
            # Warning 3: High spending volatility
            expense_transactions = df[df['amount'] < 0]['amount'].abs()
            if not expense_transactions.empty:
                cv = expense_transactions.std() / expense_transactions.mean()
                if cv > 1.5:
                    warnings.append(
                        "WARNING: Highly volatile spending patterns detected. "
                        "Consider creating a more structured budget"
                    )
            
            # Warning 4: Large transaction spike
            if len(expense_transactions) > 10:
                median_expense = expense_transactions.median()
                large_transactions = expense_transactions[expense_transactions > median_expense * 5]
                
                if len(large_transactions) > len(expense_transactions) * 0.1:
                    warnings.append(
                        f"WARNING: {len(large_transactions)} unusually large transactions detected. "
                        f"Review for unauthorized or impulsive purchases"
                    )
            
            # Warning 5: Increasing spending trend
            if len(df) > 60:  # If we have enough data
                recent_month = df[df['date'] >= df['date'].max() - timedelta(days=30)]
                previous_month = df[(df['date'] >= df['date'].max() - timedelta(days=60)) & 
                                  (df['date'] < df['date'].max() - timedelta(days=30))]
                
                if not recent_month.empty and not previous_month.empty:
                    recent_expenses = recent_month[recent_month['amount'] < 0]['amount'].abs().sum()
                    previous_expenses = previous_month[previous_month['amount'] < 0]['amount'].abs().sum()
                    
                    if recent_expenses > previous_expenses * 1.3:
                        increase = ((recent_expenses - previous_expenses) / previous_expenses) * 100
                        warnings.append(
                            f"WARNING: Spending increased {increase:.1f}% from last month. "
                            f"Monitor spending closely"
                        )
            
            # Warning 6: No emergency fund indicators
            monthly_income = user_data.get('monthly_income', 0)
            savings_goal = user_data.get('savings_goal', 0)
            
            if monthly_income > 0 and savings_goal < monthly_income * 0.1:
                warnings.append(
                    "WARNING: Low or no emergency fund target. "
                    "Set aside 3-6 months of expenses for emergencies"
                )
                
        except Exception as e:
            logger.error(f"Error generating warnings: {str(e)}")
            
        return warnings[:4]  # Top 4 most critical warnings
    
    def _identify_quick_wins(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Identify quick wins for immediate savings"""
        quick_wins = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            
            # Small frequent expenses
            if 'description' in expenses.columns:
                daily_small_expenses = expenses[
                    (expenses['amount'] < 20) & 
                    (expenses['amount'] > 3)
                ].groupby('description')['amount'].agg(['sum', 'count'])
                
                for merchant, data in daily_small_expenses.iterrows():
                    if data['count'] >= 15:  # 15+ small transactions
                        monthly_impact = data['sum']
                        quick_wins.append(
                            f"Reduce {merchant} visits: ${monthly_impact:.2f}/month potential savings"
                        )
            
            # Subscription cancellations
            subscription_candidates = self._detect_subscriptions(expenses)
            quick_wins.extend([f"QUICK WIN: {tip}" for tip in subscription_candidates[:2]])
            
            # Round-up savings
            if len(expenses) > 20:
                total_roundup = sum((np.ceil(amount) - amount) for amount in expenses['amount'])
                quick_wins.append(
                    f"Enable round-up savings: Could save ${total_roundup:.2f}/month automatically"
                )
            
            # Weekend spending reduction
            weekend_expenses = expenses[expenses['date'].dt.weekday >= 5]
            if not weekend_expenses.empty:
                weekend_total = weekend_expenses['amount'].sum()
                if weekend_total > expenses['amount'].sum() * 0.3:
                    quick_wins.append(
                        f"Plan weekend activities: ${weekend_total * 0.2:.2f}/month savings potential"
                    )
            
            # Bulk buying opportunities
            if 'category' in expenses.columns:
                grocery_expenses = expenses[expenses['category'] == 'Food']
                if len(grocery_expenses) > 20:
                    avg_grocery_trip = grocery_expenses['amount'].mean()
                    quick_wins.append(
                        f"Bulk grocery shopping: Reduce trip frequency to save ~${avg_grocery_trip * 0.1:.2f}/month"
                    )
                    
        except Exception as e:
            logger.error(f"Error identifying quick wins: {str(e)}")
            
        return quick_wins[:5]
    
    def _suggest_long_term_strategies(self, df: pd.DataFrame, user_data: Dict) -> List[str]:
        """Suggest long-term financial strategies"""
        strategies = []
        
        try:
            expenses = df[df['amount'] < 0]['amount'].abs().sum()
            income = df[df['amount'] > 0]['amount'].sum()
            
            if income > 0:
                savings_rate = (income - expenses) / income
                monthly_income = user_data.get('monthly_income', income)
                
                # Strategy 1: Increase income
                if savings_rate < self.thresholds['savings_rate_good']:
                    strategies.append(
                        "Long-term: Focus on increasing income through skills development, "
                        "side hustles, or career advancement"
                    )
                
                # Strategy 2: Automate finances
                strategies.append(
                    "Automate your finances: Set up automatic transfers to savings "
                    "and investment accounts"
                )
                
                # Strategy 3: Tax optimization
                if monthly_income > 4000:
                    strategies.append(
                        "Optimize tax strategy: Maximize retirement contributions, "
                        "HSA, and other tax-advantaged accounts"
                    )
                
                # Strategy 4: Real estate considerations
                if savings_rate >= self.thresholds['savings_rate_good']:
                    strategies.append(
                        "Consider real estate investment or homeownership as wealth-building strategy"
                    )
                
                # Strategy 5: Financial education
                strategies.append(
                    "Invest in financial education: Books, courses, or financial advisor consultation"
                )
                
                # Strategy 6: Insurance review
                strategies.append(
                    "Review insurance coverage: Health, disability, and life insurance adequacy"
                )
                
                # Strategy 7: Estate planning
                if monthly_income > 6000:
                    strategies.append(
                        "Begin estate planning: Will, beneficiaries, and power of attorney documents"
                    )
                    
        except Exception as e:
            logger.error(f"Error suggesting long-term strategies: {str(e)}")
            
        return strategies[:5]
    
    def _get_category_savings_tips(self, category: str, total: float, expenses: pd.DataFrame) -> List[str]:
        """Get specific savings tips for each category"""
        tips = []
        
        category_expenses = expenses[expenses['category'] == category]
        transaction_count = len(category_expenses)
        avg_transaction = total / max(1, transaction_count)
        
        if category == 'Food':
            if total > 800:
                tips.append(f"Food spending is high (${total:.2f}). Try meal planning and bulk cooking")
            if transaction_count > 40:
                tips.append("Frequent food purchases detected. Consider grocery shopping weekly")
                
        elif category == 'Transportation':
            if total > 500:
                tips.append(f"Transportation costs (${total:.2f}) could be reduced with carpooling")
            if avg_transaction < 15:
                tips.append("Many small transport expenses. Consider monthly transit pass")
                
        elif category == 'Entertainment':
            if total > 400:
                tips.append(f"Entertainment spending (${total:.2f}) is high. Look for free alternatives")
            if transaction_count > 20:
                tips.append("Frequent entertainment purchases. Set a monthly entertainment budget")
                
        elif category == 'Shopping':
            if total > 600:
                tips.append(f"Shopping expenses (${total:.2f}) suggest impulse buying. Try 24-hour rule")
            if transaction_count > 30:
                tips.append("Frequent shopping trips. Make lists and stick to them")
                
        elif category == 'Utilities':
            if total > 300:
                tips.append(f"Utility costs (${total:.2f}) are high. Review energy efficiency")
                
        return tips
    
    def _detect_subscriptions(self, expenses: pd.DataFrame) -> List[str]:
        """Detect potential subscription services"""
        subscription_tips = []
        
        try:
            if 'description' in expenses.columns:
                # Look for recurring charges
                merchant_amounts = expenses.groupby('description')['amount'].agg(['mean', 'std', 'count'])
                
                # Potential subscriptions: low std deviation, regular amounts, multiple occurrences
                potential_subscriptions = merchant_amounts[
                    (merchant_amounts['std'] < merchant_amounts['mean'] * 0.1) &
                    (merchant_amounts['count'] >= 3) &
                    (merchant_amounts['mean'] > 5)
                ]
                
                for merchant, data in potential_subscriptions.head(3).iterrows():
                    monthly_cost = data['mean'] * (data['count'] / max(1, 
                        (expenses['date'].max() - expenses['date'].min()).days)) * 30
                    subscription_tips.append(
                        f"Review {merchant} subscription: ~${monthly_cost:.2f}/month"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting subscriptions: {str(e)}")
            
        return subscription_tips
    
    def _generate_cash_flow_tips(self, df: pd.DataFrame) -> List[str]:
        """Generate cash flow management tips"""
        tips = []
        
        try:
            # Analyze timing of income vs expenses
            df['day_of_month'] = df['date'].dt.day
            
            income_days = df[df['amount'] > 0]['day_of_month']
            expense_days = df[df['amount'] < 0]['day_of_month']
            
            if not income_days.empty and not expense_days.empty:
                avg_income_day = income_days.mean()
                avg_expense_day = expense_days.mean()
                
                if avg_expense_day < avg_income_day - 5:
                    tips.append(
                        "Consider timing large expenses after income arrives to improve cash flow"
                    )
                
                # Check for end-of-month spending spikes
                end_month_expenses = expense_days[expense_days > 25].count()
                if end_month_expenses > len(expense_days) * 0.4:
                    tips.append(
                        "High end-of-month spending detected. Spread expenses throughout the month"
                    )
                    
        except Exception as e:
            logger.error(f"Error generating cash flow tips: {str(e)}")
            
        return tips
    
    def _identify_high_impact_savings(self, df: pd.DataFrame) -> List[str]:
        """Identify high-impact savings opportunities"""
        high_impact = []
        
        try:
            expenses = df[df['amount'] < 0].copy()
            expenses['amount'] = expenses['amount'].abs()
            
            # Find largest expense categories
            if 'category' in expenses.columns:
                category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
                
                # Top category optimization
                top_category = category_totals.index[0]
                top_amount = category_totals.iloc[0]
                
                potential_savings = top_amount * 0.15  # 15% reduction
                high_impact.append(
                    f"Reduce {top_category} spending by 15%: ${potential_savings:.2f}/month savings"
                )
            
            # Large individual transactions
            large_transactions = expenses[expenses['amount'] > expenses['amount'].quantile(0.9)]
            if not large_transactions.empty:
                large_total = large_transactions['amount'].sum()
                high_impact.append(
                    f"Review {len(large_transactions)} largest transactions: "
                    f"${large_total:.2f} total, high savings potential"
                )
                
        except Exception as e:
            logger.error(f"Error identifying high-impact savings: {str(e)}")
            
        return high_impact
    
    def _analyze_debt_patterns(self, df: pd.DataFrame) -> List[str]:
        """Analyze potential debt patterns and provide advice"""
        debt_advice = []
        
        try:
            # Look for potential interest charges, fees, or credit card payments
            if 'description' in df.columns:
                debt_keywords = ['interest', 'fee', 'credit card', 'loan', 'payment', 'finance charge']
                
                potential_debt_transactions = df[
                    df['description'].str.lower().str.contains('|'.join(debt_keywords), na=False)
                ]
                
                if not potential_debt_transactions.empty:
                    debt_total = potential_debt_transactions[potential_debt_transactions['amount'] < 0]['amount'].abs().sum()
                    
                    if debt_total > 0:
                        debt_advice.append(
                            f"Potential debt payments detected: ${debt_total:.2f}. "
                            f"Focus on high-interest debt first"
                        )
                        debt_advice.append(
                            "Consider debt consolidation or balance transfer for lower rates"
                        )
                        
        except Exception as e:
            logger.error(f"Error analyzing debt patterns: {str(e)}")
            
        return debt_advice
