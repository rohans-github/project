"""
Data Management Module
Handles data loading, processing, validation, and export
"""

import pandas as pd
import numpy as np
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
import io
from pathlib import Path

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx']
        self.required_columns = ['date', 'amount']
        self.optional_columns = ['description', 'category', 'merchant', 'account']
        
    def load_transactions_from_csv(self, file_path_or_buffer: Union[str, io.StringIO]) -> List[Dict]:
        """
        Load transactions from CSV file or buffer
        """
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(
                file_path_or_buffer,
                parse_dates=['date'] if 'date' in pd.read_csv(file_path_or_buffer, nrows=0).columns else False,
                dtype={'amount': float} if 'amount' in pd.read_csv(file_path_or_buffer, nrows=0).columns else None
            )
            
            # Validate and clean data
            df = self._validate_and_clean_data(df)
            
            # Convert to list of dictionaries
            transactions = df.to_dict('records')
            
            # Ensure proper data types
            for transaction in transactions:
                if 'date' in transaction and not isinstance(transaction['date'], str):
                    transaction['date'] = transaction['date'].isoformat() if hasattr(transaction['date'], 'isoformat') else str(transaction['date'])
                if 'amount' in transaction:
                    transaction['amount'] = float(transaction['amount'])
                    
            logger.info(f"Successfully loaded {len(transactions)} transactions from CSV")
            return transactions
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise ValueError(f"Failed to load CSV data: {str(e)}")
    
    def load_transactions_from_json(self, file_path: str) -> List[Dict]:
        """
        Load transactions from JSON file
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            # Handle different JSON structures
            if isinstance(data, list):
                transactions = data
            elif isinstance(data, dict) and 'transactions' in data:
                transactions = data['transactions']
            else:
                raise ValueError("Invalid JSON structure. Expected list or object with 'transactions' key")
            
            # Validate data
            validated_transactions = []
            for transaction in transactions:
                if self._validate_transaction(transaction):
                    validated_transactions.append(transaction)
                    
            logger.info(f"Successfully loaded {len(validated_transactions)} transactions from JSON")
            return validated_transactions
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise ValueError(f"Failed to load JSON data: {str(e)}")
    
    def load_bank_statement(self, file_path: str, bank_format: str = 'generic') -> List[Dict]:
        """
        Load transactions from bank statement with specific format handling
        """
        try:
            if bank_format == 'chase':
                return self._load_chase_format(file_path)
            elif bank_format == 'bank_of_america':
                return self._load_boa_format(file_path)
            elif bank_format == 'wells_fargo':
                return self._load_wells_fargo_format(file_path)
            else:
                return self._load_generic_bank_format(file_path)
                
        except Exception as e:
            logger.error(f"Error loading bank statement: {str(e)}")
            raise ValueError(f"Failed to load bank statement: {str(e)}")
    
    def export_to_csv(self, transactions: List[Dict], file_path: Optional[str] = None) -> str:
        """
        Export transactions to CSV format
        """
        try:
            df = pd.DataFrame(transactions)
            
            # Ensure consistent column order
            column_order = ['date', 'amount', 'description', 'category', 'merchant', 'account']
            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]
            
            if file_path:
                df.to_csv(file_path, index=False)
                logger.info(f"Exported {len(transactions)} transactions to {file_path}")
                return file_path
            else:
                # Return CSV string
                return df.to_csv(index=False)
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise ValueError(f"Failed to export to CSV: {str(e)}")
    
    def export_to_json(self, transactions: List[Dict], file_path: str) -> str:
        """
        Export transactions to JSON format
        """
        try:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'transaction_count': len(transactions),
                'transactions': transactions
            }
            
            with open(file_path, 'w') as file:
                json.dump(export_data, file, indent=2, default=str)
                
            logger.info(f"Exported {len(transactions)} transactions to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise ValueError(f"Failed to export to JSON: {str(e)}")
    
    def merge_transaction_sources(self, *transaction_lists: List[Dict]) -> List[Dict]:
        """
        Merge multiple transaction sources and remove duplicates
        """
        try:
            all_transactions = []
            for transaction_list in transaction_lists:
                all_transactions.extend(transaction_list)
            
            # Convert to DataFrame for easier duplicate handling
            df = pd.DataFrame(all_transactions)
            
            if df.empty:
                return []
            
            # Remove duplicates based on date, amount, and description
            duplicate_columns = ['date', 'amount']
            if 'description' in df.columns:
                duplicate_columns.append('description')
                
            df = df.drop_duplicates(subset=duplicate_columns, keep='first')
            
            # Sort by date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            merged_transactions = df.to_dict('records')
            logger.info(f"Merged {len(merged_transactions)} unique transactions from {len(transaction_lists)} sources")
            
            return merged_transactions
            
        except Exception as e:
            logger.error(f"Error merging transaction sources: {str(e)}")
            raise ValueError(f"Failed to merge transaction sources: {str(e)}")
    
    def categorize_transactions(self, transactions: List[Dict], custom_rules: Optional[Dict] = None) -> List[Dict]:
        """
        Automatically categorize transactions based on description
        """
        try:
            categorization_rules = self._get_default_categorization_rules()
            
            if custom_rules:
                categorization_rules.update(custom_rules)
            
            categorized_transactions = []
            
            for transaction in transactions:
                transaction_copy = transaction.copy()
                
                # Skip if already categorized
                if 'category' in transaction_copy and transaction_copy['category']:
                    categorized_transactions.append(transaction_copy)
                    continue
                
                # Categorize based on description
                description = transaction_copy.get('description', '').lower()
                category = 'Other'  # Default category
                
                for cat, keywords in categorization_rules.items():
                    if any(keyword in description for keyword in keywords):
                        category = cat
                        break
                
                transaction_copy['category'] = category
                categorized_transactions.append(transaction_copy)
            
            logger.info(f"Categorized {len(categorized_transactions)} transactions")
            return categorized_transactions
            
        except Exception as e:
            logger.error(f"Error categorizing transactions: {str(e)}")
            return transactions  # Return original if categorization fails
    
    def validate_data_quality(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics
        """
        try:
            if not transactions:
                return {'quality_score': 0, 'issues': ['No transactions provided']}
            
            df = pd.DataFrame(transactions)
            total_transactions = len(df)
            issues = []
            quality_score = 100  # Start with perfect score
            
            # Check for required columns
            missing_required = [col for col in self.required_columns if col not in df.columns]
            if missing_required:
                issues.append(f"Missing required columns: {missing_required}")
                quality_score -= 30
            
            # Check for missing values
            if 'amount' in df.columns:
                missing_amounts = df['amount'].isna().sum()
                if missing_amounts > 0:
                    issues.append(f"{missing_amounts} transactions missing amount")
                    quality_score -= (missing_amounts / total_transactions) * 20
            
            if 'date' in df.columns:
                missing_dates = df['date'].isna().sum()
                if missing_dates > 0:
                    issues.append(f"{missing_dates} transactions missing date")
                    quality_score -= (missing_dates / total_transactions) * 20
            
            # Check for invalid amounts
            if 'amount' in df.columns:
                invalid_amounts = df[df['amount'] == 0].shape[0]
                if invalid_amounts > total_transactions * 0.1:  # More than 10% zero amounts
                    issues.append(f"High number of zero-amount transactions: {invalid_amounts}")
                    quality_score -= 10
            
            # Check date range
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    date_range = (df['date'].max() - df['date'].min()).days
                    if date_range < 7:
                        issues.append("Very short date range (less than 1 week)")
                        quality_score -= 10
                except:
                    issues.append("Invalid date format detected")
                    quality_score -= 15
            
            # Check for duplicates
            if len(df.columns) >= 2:
                duplicate_rows = df.duplicated().sum()
                if duplicate_rows > 0:
                    issues.append(f"{duplicate_rows} duplicate transactions found")
                    quality_score -= (duplicate_rows / total_transactions) * 15
            
            # Check description quality
            if 'description' in df.columns:
                missing_descriptions = df['description'].isna().sum()
                empty_descriptions = (df['description'] == '').sum()
                poor_descriptions = missing_descriptions + empty_descriptions
                
                if poor_descriptions > total_transactions * 0.3:  # More than 30%
                    issues.append(f"Poor description quality: {poor_descriptions} missing/empty")
                    quality_score -= 10
            
            quality_score = max(0, quality_score)  # Ensure non-negative
            
            return {
                'quality_score': quality_score,
                'total_transactions': total_transactions,
                'issues': issues,
                'recommendations': self._get_quality_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {'quality_score': 0, 'issues': [f'Validation error: {str(e)}']}
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean transaction data
        """
        try:
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and convert data types
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])  # Remove rows with invalid dates
            
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df = df.dropna(subset=['amount'])  # Remove rows with invalid amounts
                df = df[df['amount'] != 0]  # Remove zero amounts
            
            # Clean text columns
            text_columns = ['description', 'category', 'merchant']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('nan', '')
            
            # Remove duplicate transactions
            df = df.drop_duplicates(subset=['date', 'amount', 'description'] if 'description' in df.columns else ['date', 'amount'])
            
            # Sort by date
            df = df.sort_values('date')
            
            logger.info(f"Cleaned data: {len(df)} transactions remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise ValueError(f"Data cleaning failed: {str(e)}")
    
    def _validate_transaction(self, transaction: Dict) -> bool:
        """
        Validate individual transaction
        """
        try:
            # Check required fields
            if 'date' not in transaction or 'amount' not in transaction:
                return False
            
            # Validate date
            if isinstance(transaction['date'], str):
                pd.to_datetime(transaction['date'])
            
            # Validate amount
            float(transaction['amount'])
            
            return True
            
        except:
            return False
    
    def _get_default_categorization_rules(self) -> Dict[str, List[str]]:
        """
        Get default categorization rules
        """
        return {
    def _get_default_categorization_rules(self) -> Dict[str, List[str]]:
        """
        Get default categorization rules
        """
        return {
            'Food': [
                'restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'food',
                'grocery', 'market', 'deli', 'bakery', 'starbucks', 'mcdonalds',
                'subway', 'chipotle', 'dunkin', 'taco bell', 'kfc', 'dominos'
            ],
            'Transportation': [
                'gas', 'fuel', 'uber', 'lyft', 'taxi', 'bus', 'train', 'metro',
                'parking', 'toll', 'car wash', 'auto', 'vehicle', 'dmv',
                'insurance auto', 'car payment', 'vehicle maintenance'
            ],
            'Entertainment': [
                'movie', 'cinema', 'theater', 'netflix', 'spotify', 'hulu',
                'disney', 'amazon prime', 'youtube', 'gaming', 'concert',
                'event', 'entertainment', 'recreation', 'amusement'
            ],
            'Shopping': [
                'amazon', 'walmart', 'target', 'costco', 'ebay', 'mall',
                'clothing', 'shoes', 'electronics', 'home depot', 'lowes',
                'best buy', 'pharmacy', 'cvs', 'walgreens'
            ],
            'Utilities': [
                'electric', 'electricity', 'water', 'gas utility', 'internet',
                'phone', 'mobile', 'cable', 'utility', 'power', 'energy',
                'verizon', 'att', 'comcast', 'spectrum'
            ],
            'Healthcare': [
                'doctor', 'medical', 'hospital', 'pharmacy', 'dental',
                'vision', 'health', 'clinic', 'insurance health',
                'copay', 'prescription', 'medicine'
            ],
            'Travel': [
                'hotel', 'flight', 'airline', 'airport', 'travel', 'vacation',
                'booking', 'expedia', 'airbnb', 'rental car', 'cruise'
            ],
            'Education': [
                'school', 'university', 'college', 'tuition', 'books',
                'education', 'course', 'training', 'certification'
            ],
            'Financial': [
                'bank', 'atm', 'fee', 'interest', 'loan', 'credit card',
                'investment', 'transfer', 'payment', 'finance charge'
            ]
        }
    
    def _load_chase_format(self, file_path: str) -> List[Dict]:
        """Load Chase bank statement format"""
        try:
            df = pd.read_csv(file_path)
            
            # Chase format: Transaction Date,Post Date,Description,Category,Type,Amount
            column_mapping = {
                'Transaction Date': 'date',
                'Description': 'description',
                'Category': 'category',
                'Amount': 'amount',
                'Type': 'transaction_type'
            }
            
            df = df.rename(columns=column_mapping)
            df = self._validate_and_clean_data(df)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error loading Chase format: {str(e)}")
            raise ValueError(f"Failed to load Chase format: {str(e)}")
    
    def _load_boa_format(self, file_path: str) -> List[Dict]:
        """Load Bank of America statement format"""
        try:
            df = pd.read_csv(file_path)
            
            # BoA format: Date,Description,Amount,Running Bal.
            column_mapping = {
                'Date': 'date',
                'Description': 'description',
                'Amount': 'amount',
                'Running Bal.': 'balance'
            }
            
            df = df.rename(columns=column_mapping)
            df = self._validate_and_clean_data(df)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error loading BoA format: {str(e)}")
            raise ValueError(f"Failed to load BoA format: {str(e)}")
    
    def _load_wells_fargo_format(self, file_path: str) -> List[Dict]:
        """Load Wells Fargo statement format"""
        try:
            df = pd.read_csv(file_path)
            
            # Wells Fargo format: Date,Amount,*,*,Description
            column_mapping = {
                'Date': 'date',
                'Amount': 'amount',
                'Description': 'description'
            }
            
            # Keep only mapped columns
            available_columns = [col for col in column_mapping.keys() if col in df.columns]
            df = df[available_columns].rename(columns=column_mapping)
            df = self._validate_and_clean_data(df)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error loading Wells Fargo format: {str(e)}")
            raise ValueError(f"Failed to load Wells Fargo format: {str(e)}")
    
    def _load_generic_bank_format(self, file_path: str) -> List[Dict]:
        """Load generic bank statement format with auto-detection"""
        try:
            df = pd.read_csv(file_path)
            
            # Try to auto-detect column mappings
            column_mapping = {}
            
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['date', 'trans date', 'transaction date']):
                    column_mapping[col] = 'date'
                elif any(word in col_lower for word in ['amount', 'debit', 'credit']):
                    column_mapping[col] = 'amount'
                elif any(word in col_lower for word in ['description', 'memo', 'payee']):
                    column_mapping[col] = 'description'
                elif any(word in col_lower for word in ['category', 'type']):
                    column_mapping[col] = 'category'
            
            if 'date' not in column_mapping.values() or 'amount' not in column_mapping.values():
                raise ValueError("Could not detect required columns (date, amount)")
            
            df = df.rename(columns=column_mapping)
            
            # Handle separate debit/credit columns
            if 'debit' in df.columns and 'credit' in df.columns:
                df['amount'] = df['credit'].fillna(0) - df['debit'].fillna(0)
                df = df.drop(['debit', 'credit'], axis=1)
            
            df = self._validate_and_clean_data(df)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error loading generic format: {str(e)}")
            raise ValueError(f"Failed to load generic format: {str(e)}")
    
    def _get_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations to improve data quality"""
        recommendations = []
        
        for issue in issues:
            if 'missing amount' in issue.lower():
                recommendations.append("Remove or fix transactions with missing amounts")
            elif 'missing date' in issue.lower():
                recommendations.append("Ensure all transactions have valid dates")
            elif 'duplicate' in issue.lower():
                recommendations.append("Review and remove duplicate transactions")
            elif 'description' in issue.lower():
                recommendations.append("Add meaningful descriptions to improve categorization")
            elif 'zero-amount' in issue.lower():
                recommendations.append("Review zero-amount transactions for validity")
            elif 'short date range' in issue.lower():
                recommendations.append("Consider importing more historical data for better analysis")
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations
    
    def create_sample_data(self, num_transactions: int = 100) -> List[Dict]:
        """
        Create sample transaction data for testing
        """
        try:
            import random
            from datetime import datetime, timedelta
            
            sample_transactions = []
            start_date = datetime.now() - timedelta(days=90)
            
            categories = ['Food', 'Transportation', 'Entertainment', 'Shopping', 'Utilities', 'Healthcare']
            merchants = {
                'Food': ['Starbucks', 'McDonalds', 'Grocery Store', 'Pizza Hut', 'Local Restaurant'],
                'Transportation': ['Gas Station', 'Uber', 'Metro Transit', 'Parking Meter'],
                'Entertainment': ['Netflix', 'Movie Theater', 'Spotify', 'Gaming Store'],
                'Shopping': ['Amazon', 'Target', 'Walmart', 'Local Store'],
                'Utilities': ['Electric Company', 'Internet Provider', 'Phone Company'],
                'Healthcare': ['Pharmacy', 'Doctor Office', 'Dental Clinic']
            }
            
            for i in range(num_transactions):
                # Generate random date within the last 90 days
                random_days = random.randint(0, 90)
                transaction_date = start_date + timedelta(days=random_days)
                
                # Choose random category and merchant
                category = random.choice(categories)
                merchant = random.choice(merchants[category])
                
                # Generate amount based on category
                amount_ranges = {
                    'Food': (5, 50),
                    'Transportation': (10, 100),
                    'Entertainment': (10, 200),
                    'Shopping': (20, 300),
                    'Utilities': (50, 200),
                    'Healthcare': (25, 500)
                }
                
                min_amount, max_amount = amount_ranges[category]
                amount = round(random.uniform(min_amount, max_amount), 2)
                
                # 95% expenses, 5% income
                if random.random() < 0.05:
                    amount = abs(amount) * random.randint(10, 50)  # Income
                    category = 'Income'
                    merchant = 'Salary/Income'
                else:
                    amount = -abs(amount)  # Expense
                
                transaction = {
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'description': merchant,
                    'category': category,
                    'merchant': merchant
                }
                
                sample_transactions.append(transaction)
            
            # Sort by date
            sample_transactions.sort(key=lambda x: x['date'])
            
            logger.info(f"Created {num_transactions} sample transactions")
            return sample_transactions
            
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            return []
    
    def backup_data(self, transactions: List[Dict], backup_dir: str = "backups") -> str:
        """
        Create a backup of transaction data
        """
        try:
            import os
            
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"transactions_backup_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Export to JSON with metadata
            backup_data = {
                'backup_date': datetime.now().isoformat(),
                'version': '1.0',
                'transaction_count': len(transactions),
                'transactions': transactions
            }
            
            with open(backup_path, 'w') as file:
                json.dump(backup_data, file, indent=2, default=str)
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise ValueError(f"Failed to create backup: {str(e)}")
    
    def restore_from_backup(self, backup_path: str) -> List[Dict]:
        """
        Restore transaction data from backup
        """
        try:
            with open(backup_path, 'r') as file:
                backup_data = json.load(file)
            
            if 'transactions' not in backup_data:
                raise ValueError("Invalid backup format")
            
            transactions = backup_data['transactions']
            logger.info(f"Restored {len(transactions)} transactions from backup")
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
            raise ValueError(f"Failed to restore from backup: {str(e)}")
    
    def get_data_summary(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the transaction data
        """
        try:
            if not transactions:
                return {'error': 'No transactions provided'}
            
            df = pd.DataFrame(transactions)
            
            # Basic statistics
            total_transactions = len(df)
            date_range = None
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                date_range = {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d'),
                    'days': (df['date'].max() - df['date'].min()).days
                }
            
            # Amount statistics
            amount_stats = {}
            if 'amount' in df.columns:
                expenses = df[df['amount'] < 0]['amount'].abs()
                income = df[df['amount'] > 0]['amount']
                
                amount_stats = {
                    'total_expenses': expenses.sum() if not expenses.empty else 0,
                    'total_income': income.sum() if not income.empty else 0,
                    'average_expense': expenses.mean() if not expenses.empty else 0,
                    'average_income': income.mean() if not income.empty else 0,
                    'largest_expense': expenses.max() if not expenses.empty else 0,
                    'largest_income': income.max() if not income.empty else 0
                }
                
                amount_stats['net_change'] = amount_stats['total_income'] - amount_stats['total_expenses']
                if amount_stats['total_income'] > 0:
                    amount_stats['savings_rate'] = (amount_stats['net_change'] / amount_stats['total_income']) * 100
            
            # Category breakdown
            category_breakdown = {}
            if 'category' in df.columns:
                category_breakdown = df.groupby('category')['amount'].agg(['count', 'sum']).to_dict()
            
            summary = {
                'total_transactions': total_transactions,
                'date_range': date_range,
                'amount_statistics': amount_stats,
                'category_breakdown': category_breakdown,
                'data_quality': self.validate_data_quality(transactions)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {'error': str(e)} '
