"""
AI-Powered Personal Finance Assistant
Main application entry point
"""

import os
import sys
from datetime import datetime
import streamlit as st
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.transaction_analyzer import TransactionAnalyzer
from src.predictive_model import PredictiveModel
from src.advice_engine import AdviceEngine
from src.data_manager import DataManager
from src.visualization import FinanceVisualizer
from utils.config import Config
from utils.logger import setup_logger

# Configure Streamlit page
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger()

class FinanceAssistantApp:
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager()
        self.transaction_analyzer = TransactionAnalyzer()
        self.predictive_model = PredictiveModel()
        self.advice_engine = AdviceEngine()
        self.visualizer = FinanceVisualizer()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
            
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("🏦 Finance Assistant")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["Dashboard", "Transaction Analysis", "Predictions", "Advice", "Data Upload", "Settings"]
        )
        
        # User profile section
        st.sidebar.markdown("### User Profile")
        monthly_income = st.sidebar.number_input("Monthly Income ($)", value=5000, min_value=0)
        savings_goal = st.sidebar.number_input("Monthly Savings Goal ($)", value=1000, min_value=0)
        
        st.session_state.user_data.update({
            'monthly_income': monthly_income,
            'savings_goal': savings_goal
        })
        
        return page
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("📊 Financial Dashboard")
        
        if not st.session_state.transactions:
            st.warning("No transaction data available. Please upload your financial data first.")
            return
            
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        transactions = st.session_state.transactions
        total_spent = sum(t.get('amount', 0) for t in transactions if t.get('amount', 0) < 0)
        total_income = sum(t.get('amount', 0) for t in transactions if t.get('amount', 0) > 0)
        
        with col1:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Total Expenses", f"${abs(total_spent):,.2f}")
        with col3:
            st.metric("Net Savings", f"${total_income + total_spent:,.2f}")
        with col4:
            savings_rate = ((total_income + total_spent) / total_income * 100) if total_income > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        # Visualizations
        st.markdown("### Spending Overview")
        fig_spending = self.visualizer.create_spending_chart(transactions)
        if fig_spending:
            st.plotly_chart(fig_spending, use_container_width=True)
            
        # Category breakdown
        col1, col2 = st.columns(2)
        with col1:
            fig_categories = self.visualizer.create_category_pie_chart(transactions)
            if fig_categories:
                st.plotly_chart(fig_categories, use_container_width=True)
        
        with col2:
            fig_trends = self.visualizer.create_trend_chart(transactions)
            if fig_trends:
                st.plotly_chart(fig_trends, use_container_width=True)
    
    def render_transaction_analysis(self):
        """Render transaction analysis page"""
        st.title("🔍 Transaction Analysis")
        
        if not st.session_state.transactions:
            st.warning("No transaction data available.")
            return
            
        # Run analysis
        if st.button("Run Analysis"):
            with st.spinner("Analyzing transactions..."):
                analysis = self.transaction_analyzer.analyze_transactions(
                    st.session_state.transactions
                )
                st.session_state.analysis_results = analysis
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Spending Patterns")
                if 'patterns' in results:
                    for pattern in results['patterns']:
                        st.write(f"• {pattern}")
                        
                st.subheader("Anomalies Detected")
                if 'anomalies' in results:
                    for anomaly in results['anomalies']:
                        st.warning(f"⚠️ {anomaly}")
            
            with col2:
                st.subheader("Category Analysis")
                if 'category_analysis' in results:
                    for category, data in results['category_analysis'].items():
                        st.write(f"**{category}**: ${data.get('total', 0):.2f}")
                        
                st.subheader("Monthly Trends")
                if 'trends' in results:
                    for trend in results['trends']:
                        st.info(f"📈 {trend}")
    
    def render_predictions(self):
        """Render predictions page"""
        st.title("🔮 Financial Predictions")
        
        if not st.session_state.transactions:
            st.warning("No transaction data available for predictions.")
            return
            
        # Generate predictions
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                predictions = self.predictive_model.generate_predictions(
                    st.session_state.transactions,
                    st.session_state.user_data
                )
                st.session_state.predictions = predictions
        
        # Display predictions
        if st.session_state.predictions:
            pred = st.session_state.predictions
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Next Month Forecast")
                if 'monthly_forecast' in pred:
                    forecast = pred['monthly_forecast']
                    st.metric("Predicted Expenses", f"${forecast.get('expenses', 0):.2f}")
                    st.metric("Predicted Savings", f"${forecast.get('savings', 0):.2f}")
                    
            with col2:
                st.subheader("Goal Achievement")
                if 'goal_prediction' in pred:
                    goal_pred = pred['goal_prediction']
                    st.write(f"Savings Goal Achievement: {goal_pred.get('probability', 0):.1%}")
                    
            # Visualization
            if 'forecast_chart' in pred:
                st.plotly_chart(pred['forecast_chart'], use_container_width=True)
    
    def render_advice(self):
        """Render personalized advice page"""
        st.title("💡 Personalized Financial Advice")
        
        if st.button("Get Personalized Advice"):
            with st.spinner("Generating advice..."):
                advice = self.advice_engine.generate_advice(
                    st.session_state.transactions,
                    st.session_state.user_data,
                    st.session_state.analysis_results
                )
                
                st.subheader("🎯 Priority Recommendations")
                for rec in advice.get('priority_recommendations', []):
                    st.success(f"✅ {rec}")
                    
                st.subheader("💰 Savings Opportunities")
                for opp in advice.get('savings_opportunities', []):
                    st.info(f"💡 {opp}")
                    
                st.subheader("⚠️ Risk Warnings")
                for warning in advice.get('warnings', []):
                    st.warning(f"⚠️ {warning}")
                    
                st.subheader("📊 Budgeting Tips")
                for tip in advice.get('budgeting_tips', []):
                    st.write(f"• {tip}")
    
    def render_data_upload(self):
        """Render data upload page"""
        st.title("📁 Data Upload")
        
        st.markdown("""
        Upload your financial data to get started. Supported formats:
        - CSV files with transaction data
        - Bank statement exports
        - Manual data entry
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your transaction data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                transactions = self.data_manager.load_transactions_from_csv(uploaded_file)
                st.session_state.transactions = transactions
                st.success(f"Loaded {len(transactions)} transactions successfully!")
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(transactions[:10])
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Manual entry
        st.subheader("Manual Transaction Entry")
        with st.form("manual_transaction"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date = st.date_input("Date")
                amount = st.number_input("Amount ($)", step=0.01)
                
            with col2:
                category = st.selectbox(
                    "Category",
                    ["Food", "Transportation", "Entertainment", "Utilities", "Healthcare", "Other"]
                )
                description = st.text_input("Description")
                
            with col3:
                transaction_type = st.selectbox("Type", ["Expense", "Income"])
                
            if st.form_submit_button("Add Transaction"):
                new_transaction = {
                    'date': date.isoformat(),
                    'amount': -amount if transaction_type == "Expense" else amount,
                    'category': category,
                    'description': description,
                    'type': transaction_type
                }
                
                if 'transactions' not in st.session_state:
                    st.session_state.transactions = []
                st.session_state.transactions.append(new_transaction)
                st.success("Transaction added successfully!")
    
    def render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        st.subheader("Application Settings")
        
        # Theme settings
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        
        # Notification settings
        st.subheader("Notifications")
        enable_alerts = st.checkbox("Enable spending alerts")
        alert_threshold = st.number_input("Alert threshold ($)", value=500)
        
        # Data settings
        st.subheader("Data Management")
        if st.button("Export Data"):
            if st.session_state.transactions:
                csv_data = self.data_manager.export_to_csv(st.session_state.transactions)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    "financial_data.csv",
                    "text/csv"
                )
            else:
                st.warning("No data to export")
                
        if st.button("Clear All Data"):
            st.session_state.transactions = []
            st.session_state.analysis_results = {}
            st.session_state.predictions = {}
            st.success("Data cleared successfully!")
    
    def run(self):
        """Run the main application"""
        self.initialize_session_state()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "Dashboard":
            self.render_dashboard()
        elif selected_page == "Transaction Analysis":
            self.render_transaction_analysis()
        elif selected_page == "Predictions":
            self.render_predictions()
        elif selected_page == "Advice":
            self.render_advice()
        elif selected_page == "Data Upload":
            self.render_data_upload()
        elif selected_page == "Settings":
            self.render_settings()

def main():
    """Main function"""
    try:
        app = FinanceAssistantApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()
