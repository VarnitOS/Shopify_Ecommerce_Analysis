"""Feature engineering for customer data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid

from sqlalchemy import func, desc, and_, or_, text
import mlflow

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import db_session
from src.models.entities import Customer, Order, Product, Review, FeatureStat

logger = get_logger(__name__)
config = get_config()


class CustomerFeatureEngineer:
    """Generate features for customer segmentation and prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_start_date = datetime.utcnow() - timedelta(days=365)  # 1 year of data
        logger.info("Initialized CustomerFeatureEngineer")
    
    def load_customer_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load customer data from the database.
        
        Args:
            limit: Optional limit on number of customers to load
            
        Returns:
            DataFrame with customer data
        """
        with db_session() as session:
            query = session.query(Customer)
            
            if limit:
                query = query.limit(limit)
            
            customers = query.all()
            
            # Convert to dataframe
            customers_data = []
            for customer in customers:
                # Basic customer attributes
                customer_dict = {
                    'customer_id': customer.id,
                    'email': customer.email,
                    'first_name': customer.first_name,
                    'last_name': customer.last_name,
                    'phone': customer.phone,
                    'verified_email': customer.verified_email,
                    'accepts_marketing': customer.accepts_marketing,
                    'created_at': customer.created_at,
                    'updated_at': customer.updated_at,
                }
                
                customers_data.append(customer_dict)
        
        df = pd.DataFrame(customers_data)
        logger.info(f"Loaded {len(df)} customers from database")
        return df
    
    def load_order_data(self, customer_ids: Optional[List[uuid.UUID]] = None) -> pd.DataFrame:
        """Load order data from the database.
        
        Args:
            customer_ids: Optional list of customer IDs to filter by
            
        Returns:
            DataFrame with order data
        """
        with db_session() as session:
            query = session.query(Order)
            
            if customer_ids:
                query = query.filter(Order.customer_id.in_(customer_ids))
            
            # Only include orders after the feature start date
            query = query.filter(Order.created_at >= self.feature_start_date)
            
            orders = query.all()
            
            # Convert to dataframe
            orders_data = []
            for order in orders:
                order_dict = {
                    'order_id': order.id,
                    'customer_id': order.customer_id,
                    'total_price': order.total_price,
                    'subtotal_price': order.subtotal_price,
                    'total_tax': order.total_tax,
                    'total_discounts': order.total_discounts,
                    'total_shipping': order.total_shipping,
                    'currency': order.currency,
                    'financial_status': order.financial_status,
                    'fulfillment_status': order.fulfillment_status,
                    'created_at': order.created_at,
                }
                
                orders_data.append(order_dict)
        
        df = pd.DataFrame(orders_data)
        logger.info(f"Loaded {len(df)} orders from database")
        return df
    
    def compute_recency_frequency_monetary(self, orders_df: pd.DataFrame,
                                          reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Compute RFM features for customers.
        
        Args:
            orders_df: DataFrame with order data
            reference_date: Reference date for recency calculation. If None, use current time
            
        Returns:
            DataFrame with RFM features
        """
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        # Ensure we have data
        if len(orders_df) == 0:
            logger.warning("No orders data available for RFM calculation")
            return pd.DataFrame(columns=['customer_id', 'recency_days', 'frequency', 'monetary'])
        
        # Compute RFM values
        rfm = orders_df.groupby('customer_id').agg({
            'created_at': lambda x: (reference_date - x.max()).total_seconds() / (60 * 60 * 24),  # Recency in days
            'order_id': 'count',  # Frequency
            'total_price': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = ['customer_id', 'recency_days', 'frequency', 'monetary']
        
        logger.info(f"Computed RFM features for {len(rfm)} customers")
        return rfm
    
    def compute_purchase_patterns(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compute purchase pattern features for customers.
        
        Args:
            orders_df: DataFrame with order data
            
        Returns:
            DataFrame with purchase pattern features
        """
        # Ensure we have data
        if len(orders_df) == 0:
            logger.warning("No orders data available for purchase pattern calculation")
            return pd.DataFrame(columns=['customer_id', 'avg_order_value', 'std_order_value', 
                                         'avg_items_per_order', 'days_between_orders'])
        
        # Compute average and standard deviation of order value
        order_stats = orders_df.groupby('customer_id').agg({
            'total_price': ['mean', 'std']
        }).reset_index()
        
        order_stats.columns = ['customer_id', 'avg_order_value', 'std_order_value']
        
        # Replace NaN values with 0
        order_stats = order_stats.fillna(0)
        
        # Compute average time between orders
        # First sort orders by customer and date
        orders_df = orders_df.sort_values(['customer_id', 'created_at'])
        
        # Calculate days between consecutive orders for each customer
        orders_df['prev_date'] = orders_df.groupby('customer_id')['created_at'].shift(1)
        orders_df['days_between'] = (orders_df['created_at'] - orders_df['prev_date']).dt.total_seconds() / (60 * 60 * 24)
        
        # Calculate average days between orders for each customer
        time_between = orders_df.groupby('customer_id').agg({
            'days_between': 'mean'
        }).reset_index()
        
        time_between.columns = ['customer_id', 'days_between_orders']
        time_between = time_between.fillna(0)  # For customers with only one order
        
        # Compute average number of items per order
        # This would require joining with order_product_association table
        # For simplicity, we'll leave this for now
        
        # Merge all features
        patterns = pd.merge(order_stats, time_between, on='customer_id', how='left')
        
        logger.info(f"Computed purchase pattern features for {len(patterns)} customers")
        return patterns
    
    def compute_customer_lifetime_value(self, rfm_df: pd.DataFrame, purchase_patterns_df: pd.DataFrame,
                                       churn_rate: float = 0.1) -> pd.DataFrame:
        """Compute customer lifetime value.
        
        Args:
            rfm_df: DataFrame with RFM features
            purchase_patterns_df: DataFrame with purchase pattern features
            churn_rate: Estimated churn rate
            
        Returns:
            DataFrame with CLV
        """
        # Ensure we have data
        if len(rfm_df) == 0 or len(purchase_patterns_df) == 0:
            logger.warning("No data available for CLV calculation")
            return pd.DataFrame(columns=['customer_id', 'customer_lifetime_value'])
        
        # Merge RFM and purchase patterns
        clv_df = pd.merge(rfm_df, purchase_patterns_df, on='customer_id', how='left')
        
        # Simple CLV calculation: (Average Order Value * Purchase Frequency) / Churn Rate
        clv_df['customer_lifetime_value'] = (clv_df['avg_order_value'] * clv_df['frequency']) / churn_rate
        
        # Filter columns
        clv_df = clv_df[['customer_id', 'customer_lifetime_value']]
        
        logger.info(f"Computed CLV for {len(clv_df)} customers")
        return clv_df
    
    def compute_all_features(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Compute all customer features.
        
        Args:
            limit: Optional limit on number of customers to process
            
        Returns:
            DataFrame with all customer features
        """
        # Load customer data
        customers_df = self.load_customer_data(limit=limit)
        
        if len(customers_df) == 0:
            logger.warning("No customers found in database")
            return pd.DataFrame()
        
        # Load order data for these customers
        orders_df = self.load_order_data(customer_ids=customers_df['customer_id'].tolist())
        
        # Compute features
        rfm_df = self.compute_recency_frequency_monetary(orders_df)
        purchase_patterns_df = self.compute_purchase_patterns(orders_df)
        clv_df = self.compute_customer_lifetime_value(rfm_df, purchase_patterns_df)
        
        # Merge all features
        features = customers_df.merge(rfm_df, on='customer_id', how='left')
        features = features.merge(purchase_patterns_df, on='customer_id', how='left')
        features = features.merge(clv_df, on='customer_id', how='left')
        
        # Fill NaN values for customers with no orders
        features = features.fillna({
            'recency_days': -1,  # -1 indicates no orders
            'frequency': 0,
            'monetary': 0,
            'avg_order_value': 0,
            'std_order_value': 0,
            'days_between_orders': -1,  # -1 indicates no or only one order
            'customer_lifetime_value': 0
        })
        
        logger.info(f"Computed all features for {len(features)} customers")
        return features
    
    def save_features_to_db(self, features_df: pd.DataFrame) -> int:
        """Save computed features to the database.
        
        Args:
            features_df: DataFrame with computed features
            
        Returns:
            Number of feature records saved
        """
        if len(features_df) == 0:
            logger.warning("No features to save to database")
            return 0
        
        # Columns to save as features
        feature_columns = [
            'recency_days', 'frequency', 'monetary', 
            'avg_order_value', 'std_order_value', 'days_between_orders',
            'customer_lifetime_value'
        ]
        
        count = 0
        with db_session() as session:
            for _, row in features_df.iterrows():
                customer_id = row['customer_id']
                
                # Create feature stats for each feature
                for feature_name in feature_columns:
                    if feature_name in row and not pd.isna(row[feature_name]):
                        # Use upsert pattern
                        stmt = insert(FeatureStat).values(
                            entity_type='customer',
                            entity_id=customer_id,
                            feature_name=feature_name,
                            stat_type='value',
                            value=float(row[feature_name]),
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['entity_type', 'entity_id', 'feature_name', 'stat_type'],
                            set_={
                                'value': float(row[feature_name]),
                                'updated_at': datetime.utcnow()
                            }
                        )
                        
                        session.execute(stmt)
                        count += 1
        
        logger.info(f"Saved {count} feature records to database")
        return count
    
    def log_features_to_mlflow(self, features_df: pd.DataFrame, experiment_name: str = "customer_features"):
        """Log feature statistics to MLflow.
        
        Args:
            features_df: DataFrame with computed features
            experiment_name: MLflow experiment name
        """
        if len(features_df) == 0:
            logger.warning("No features to log to MLflow")
            return
        
        # Set MLflow tracking URI
        mlflow_uri = config.get('mlops.mlflow.tracking_uri')
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name=f"customer_features_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            # Log feature statistics
            feature_columns = [
                'recency_days', 'frequency', 'monetary', 
                'avg_order_value', 'std_order_value', 'days_between_orders',
                'customer_lifetime_value'
            ]
            
            for col in feature_columns:
                if col in features_df.columns:
                    stats = features_df[col].describe()
                    
                    # Log statistics as metrics
                    mlflow.log_metric(f"{col}_mean", stats['mean'])
                    mlflow.log_metric(f"{col}_std", stats['std'])
                    mlflow.log_metric(f"{col}_min", stats['min'])
                    mlflow.log_metric(f"{col}_25%", stats['25%'])
                    mlflow.log_metric(f"{col}_50%", stats['50%'])
                    mlflow.log_metric(f"{col}_75%", stats['75%'])
                    mlflow.log_metric(f"{col}_max", stats['max'])
            
            # Log feature correlations
            corr = features_df[feature_columns].corr()
            
            # Save correlation matrix as artifact
            corr_file = "feature_correlations.csv"
            corr.to_csv(corr_file)
            mlflow.log_artifact(corr_file)
            
            # Log number of customers
            mlflow.log_metric("num_customers", len(features_df))
            
            # Log run metadata
            mlflow.log_param("feature_start_date", self.feature_start_date.isoformat())
            mlflow.log_param("run_date", datetime.utcnow().isoformat())
        
        logger.info(f"Logged feature statistics to MLflow experiment: {experiment_name}")


def run_feature_engineering(limit: Optional[int] = None, save_to_db: bool = True, 
                          log_to_mlflow: bool = True) -> pd.DataFrame:
    """Run the customer feature engineering process.
    
    Args:
        limit: Optional limit on number of customers to process
        save_to_db: Whether to save features to database
        log_to_mlflow: Whether to log feature statistics to MLflow
        
    Returns:
        DataFrame with computed features
    """
    engineer = CustomerFeatureEngineer()
    
    # Compute features
    features_df = engineer.compute_all_features(limit=limit)
    
    # Save features to database if requested
    if save_to_db:
        engineer.save_features_to_db(features_df)
    
    # Log features to MLflow if requested
    if log_to_mlflow:
        engineer.log_features_to_mlflow(features_df)
    
    return features_df


if __name__ == "__main__":
    """Run the feature engineering process when executed directly."""
    run_feature_engineering() 