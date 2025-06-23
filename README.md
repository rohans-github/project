# AI-Powered Personal Finance Assistant 💰

A comprehensive AI-powered personal finance assistant that combines transaction analysis, predictive modeling, and personalized financial advice in a single, accessible platform.

## 🚀 Features

### Core Functionality
- **Transaction Analysis**: Automatically categorize and analyze spending patterns
- **Predictive Modeling**: Forecast future expenses, income, and savings using machine learning
- **AI-Powered Advice**: Get personalized financial recommendations based on your spending habits
- **Interactive Visualizations**: Beautiful charts and graphs to understand your financial data
- **Data Import/Export**: Support for multiple file formats (CSV, JSON, Excel)
- **Bank Statement Integration**: Compatible with major bank formats

### Advanced Features
- **Anomaly Detection**: Identify unusual transactions and spending patterns
- **Budget Tracking**: Compare actual spending against budgets
- **Goal Management**: Track progress toward savings and financial goals
- **Risk Assessment**: Evaluate financial risks and vulnerabilities
- **Cash Flow Forecasting**: Predict future cash flow patterns
- **Spending Optimization**: Identify opportunities to reduce expenses

## 📊 Dashboard Features

- **Real-time Financial Overview**: Key metrics and KPIs at a glance
- **Category Breakdown**: Detailed analysis of spending by category
- **Trend Analysis**: Visualize spending trends over time
- **Savings Rate Tracking**: Monitor your savings progress
- **Monthly/Weekly Patterns**: Understand your spending habits

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-finance-assistant.git
   cd ai-finance-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Alternative Installation (Virtual Environment)

```bash
# Create virtual environment
python -m venv finance_env

# Activate virtual environment
# On Windows:
finance_env\Scripts\activate
# On macOS/Linux:
source finance_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 📁 Project Structure

```
ai-finance-assistant/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── config.json            # Application configuration
├── README.md              # Project documentation
├── src/                   # Source code modules
│   ├── transaction_analyzer.py    # Transaction analysis engine
│   ├── predictive_model.py        # Machine learning models
│   ├── advice_engine.py           # AI advice generation
│   ├── data_manager.py            # Data handling and processing
│   └── visualization.py           # Chart and graph creation
├── utils/                 # Utility modules
│   ├── config.py          # Configuration management
│   └── logger.py          # Logging setup
├── data/                  # Data storage (created automatically)
├── logs/                  # Application logs (created automatically)
└── backups/               # Data backups (created automatically)
```

## 🔧 Configuration

The application uses a `config.json` file for configuration. Key settings include:

```json
{
  "app": {
    "name": "AI Personal Finance Assistant",
    "version": "1.0.0",
    "debug": false,
    "port": 8501
  },
  "analysis": {
    "anomaly_detection_enabled": true,
    "anomaly_threshold": 2.5,
    "prediction_horizon_months": 6
  },
  "notifications": {
    "spending_alerts": true,
    "budget_warnings": true,
    "goal_reminders": true
  }
}
```

## 📊 Data Formats

### Supported Input Formats

#### CSV Format
```csv
date,amount,description,category
2024-01-15,-45.67,Grocery Store,Food
2024-01-15,-12.50,Coffee Shop,Food
2024-01-16,3500.00,Salary,Income
```

#### JSON Format
```json
{
  "transactions": [
    {
      "date": "2024-01-15",
      "amount": -45.67,
      "description": "Grocery Store",
      "category": "Food"
    }
  ]
}
```

### Bank Statement Formats
- Chase Bank
- Bank of America
- Wells Fargo
- Generic CSV formats

## 🤖 AI Features

### Machine Learning Models
- **Linear Regression**: For trend analysis and basic forecasting
- **Random Forest**: For complex pattern recognition
- **Anomaly Detection**: Using statistical methods and clustering
- **Time Series Analysis**: For seasonal pattern detection

### AI-Powered Insights
- Spending pattern recognition
- Budget optimization suggestions
- Goal achievement predictions
- Risk assessment and warnings
- Personalized financial advice

## 📈 Usage Examples

### 1. Upload Transaction Data
```python
# Navigate to "Data Upload" page
# Select your CSV file or enter transactions manually
# The system automatically categorizes transactions
```

### 2. Analyze Spending Patterns
```python
# Go to "Transaction Analysis" page
# Click "Run Analysis" to get insights about:
# - Spending patterns by day/category
# - Anomalous transactions
# - Monthly trends
```

### 3. Generate Predictions
```python
# Visit "Predictions" page
# Click "Generate Predictions" to see:
# - Next month's forecasted expenses
# - Savings goal achievement probability
# - Cash flow predictions
```

### 4. Get Personalized Advice
```python
# Open "Advice" page
# Click "Get Personalized Advice" for:
# - Priority recommendations
# - Savings opportunities
# - Budgeting tips
# - Investment suggestions
```

## 🎯 Key Metrics Tracked

- **Savings Rate**: Percentage of income saved
- **Expense Ratios**: Spending by category as % of income
- **Cash Flow**: Monthly income vs expenses
- **Budget Variance**: Actual vs planned spending
- **Goal Progress**: Tracking toward financial goals
- **Risk Score**: Overall financial health assessment

## 🔒 Security & Privacy

- **Local Data Storage**: All data stays on your device
- **No Cloud Dependencies**: Works completely offline
- **Data Encryption**: Optional encryption for sensitive data
- **Backup & Recovery**: Automatic data backups
- **Audit Logging**: Track all data operations

## 🛡️ Error Handling

The application includes comprehensive error handling:
- Data validation and cleaning
- Graceful handling of malformed files
- User-friendly error messages
- Automatic recovery mechanisms
- Detailed logging for troubleshooting

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_transaction_analyzer.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/ utils/

# Run linting
flake8 src/ utils/

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**Issue**: "Module not found" error
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Port already in use
**Solution**: Change the port in config.json or use: `streamlit run main.py --server.port 8502`

**Issue**: File upload fails
**Solution**: Check file format and size limits in the configuration

### Getting Help

1. Check the [Issues](https://github.com/yourusername/ai-finance-assistant/issues) page
2. Review the logs in the `logs/` directory
3. Create a new issue with details about your problem

## 🔮 Future Enhancements

- [ ] Mobile app version
- [ ] Real-time bank API integration
- [ ] Advanced investment tracking
- [ ] Multi-currency support
- [ ] Family/shared account management
- [ ] Advanced ML models (neural networks)
- [ ] Natural language query interface
- [ ] Integration with popular financial services

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [Plotly](https://plotly.com/python/) for interactive visualizations
- Powered by [scikit-learn](https://scikit-learn.org/) for machine learning
- Data processing with [Pandas](https://pandas.pydata.org/)

## 📊 Screenshots

### Dashboard Overview
![Dashboard](docs/images/dashboard.png)

### Transaction Analysis
![Analysis](docs/images/analysis.png)

### Predictions & Forecasting
![Predictions](docs/images/predictions.png)

---

**Made with ❤️ for better financial health**

For questions or suggestions, please open an issue or contact [your-email@example.com](mailto:your-email@example.com)
