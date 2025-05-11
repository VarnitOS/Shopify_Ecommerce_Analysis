"""ETL pipeline for loading Shopify data into the database."""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import and_, or_

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import db_session
from src.models.entities import Customer, Product, Category, Order, Review
from .shopify_client import ShopifyAPIClient


logger = get_logger(__name__)
config = get_config()


class ShopifyETL:
    """ETL process for loading Shopify data into the database."""
    
    def __init__(self, shopify_client: Optional[ShopifyAPIClient] = None):
        """Initialize the ETL process.
        
        Args:
            shopify_client: Shopify API client. If None, create a new one
        """
        self.shopify_client = shopify_client or ShopifyAPIClient()
        logger.info("Initialized Shopify ETL process")
    
    def _save_raw_data(self, data: Dict[str, List[Dict]], data_dir: str = 'data/raw') -> Dict[str, str]:
        """Save raw API data to JSON files.
        
        Args:
            data: Dictionary containing lists of products, orders, and customers
            data_dir: Directory to save the data files
            
        Returns:
            Dictionary with file paths
        """
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_paths = {}
        
        for data_type, items in data.items():
            if not items:
                continue
                
            file_name = f"{data_type}_{timestamp}.json"
            file_path = os.path.join(data_dir, file_name)
            
            with open(file_path, 'w') as f:
                json.dump(items, f, indent=2)
            
            file_paths[data_type] = file_path
            logger.info(f"Saved {len(items)} {data_type} to {file_path}")
        
        return file_paths
    
    def _transform_customer(self, customer_data: Dict) -> Dict:
        """Transform customer data from Shopify API format to our model.
        
        Args:
            customer_data: Customer data from Shopify API
            
        Returns:
            Transformed customer data
        """
        return {
            'shopify_id': str(customer_data.get('id')),
            'email': customer_data.get('email'),
            'first_name': customer_data.get('first_name'),
            'last_name': customer_data.get('last_name'),
            'phone': customer_data.get('phone'),
            'verified_email': customer_data.get('verified_email', False),
            'accepts_marketing': customer_data.get('accepts_marketing', False),
            'created_at': datetime.fromisoformat(customer_data.get('created_at').replace('Z', '+00:00')),
            'updated_at': datetime.fromisoformat(customer_data.get('updated_at').replace('Z', '+00:00')),
            'metadata': {k: v for k, v in customer_data.items() if k not in [
                'id', 'email', 'first_name', 'last_name', 'phone', 
                'verified_email', 'accepts_marketing', 'created_at', 'updated_at'
            ]}
        }
    
    def _transform_product(self, product_data: Dict) -> Dict:
        """Transform product data from Shopify API format to our model.
        
        Args:
            product_data: Product data from Shopify API
            
        Returns:
            Transformed product data
        """
        variants = product_data.get('variants', [])
        price = float(variants[0].get('price', 0)) if variants else 0
        compare_price = float(variants[0].get('compare_at_price', 0)) if variants and variants[0].get('compare_at_price') else None
        sku = variants[0].get('sku') if variants else None
        inventory = sum(int(v.get('inventory_quantity', 0)) for v in variants)
        
        published_at = None
        if product_data.get('published_at'):
            published_at = datetime.fromisoformat(product_data.get('published_at').replace('Z', '+00:00'))
        
        return {
            'shopify_id': str(product_data.get('id')),
            'title': product_data.get('title'),
            'description': product_data.get('body_html'),
            'vendor': product_data.get('vendor'),
            'product_type': product_data.get('product_type'),
            'price': price,
            'compare_at_price': compare_price,
            'sku': sku,
            'inventory_quantity': inventory,
            'requires_shipping': variants[0].get('requires_shipping', True) if variants else True,
            'taxable': variants[0].get('taxable', True) if variants else True,
            'published_at': published_at,
            'created_at': datetime.fromisoformat(product_data.get('created_at').replace('Z', '+00:00')),
            'updated_at': datetime.fromisoformat(product_data.get('updated_at').replace('Z', '+00:00')),
            'metadata': {
                'handle': product_data.get('handle'),
                'images': product_data.get('images'),
                'tags': product_data.get('tags'),
                'variants': product_data.get('variants'),
                'options': product_data.get('options')
            }
        }
    
    def _transform_order(self, order_data: Dict) -> Dict:
        """Transform order data from Shopify API format to our model.
        
        Args:
            order_data: Order data from Shopify API
            
        Returns:
            Transformed order data
        """
        processed_at = None
        if order_data.get('processed_at'):
            processed_at = datetime.fromisoformat(order_data.get('processed_at').replace('Z', '+00:00'))
            
        cancelled_at = None
        if order_data.get('cancelled_at'):
            cancelled_at = datetime.fromisoformat(order_data.get('cancelled_at').replace('Z', '+00:00'))
        
        return {
            'shopify_id': str(order_data.get('id')),
            'order_number': str(order_data.get('order_number')),
            'email': order_data.get('email'),
            'total_price': float(order_data.get('total_price', 0)),
            'subtotal_price': float(order_data.get('subtotal_price', 0)),
            'total_tax': float(order_data.get('total_tax', 0)),
            'total_discounts': float(order_data.get('total_discounts', 0)),
            'total_shipping': float(order_data.get('shipping_lines', [{}])[0].get('price', 0)) if order_data.get('shipping_lines') else 0,
            'currency': order_data.get('currency', 'USD'),
            'financial_status': order_data.get('financial_status'),
            'fulfillment_status': order_data.get('fulfillment_status'),
            'payment_method': order_data.get('payment_gateway_names', [])[0] if order_data.get('payment_gateway_names') else None,
            'order_status_url': order_data.get('order_status_url'),
            'created_at': datetime.fromisoformat(order_data.get('created_at').replace('Z', '+00:00')),
            'updated_at': datetime.fromisoformat(order_data.get('updated_at').replace('Z', '+00:00')),
            'processed_at': processed_at,
            'cancelled_at': cancelled_at,
            'metadata': {
                'line_items': order_data.get('line_items'),
                'shipping_address': order_data.get('shipping_address'),
                'billing_address': order_data.get('billing_address'),
                'discount_codes': order_data.get('discount_codes'),
                'note': order_data.get('note'),
                'tags': order_data.get('tags')
            }
        }
    
    def _load_customers(self, customers_data: List[Dict]) -> int:
        """Load customer data into the database.
        
        Args:
            customers_data: List of customer data from Shopify API
            
        Returns:
            Number of customers processed
        """
        count = 0
        
        with db_session() as session:
            for customer in customers_data:
                transformed = self._transform_customer(customer)
                
                # Use upsert pattern (insert or update)
                stmt = insert(Customer).values(**transformed)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['shopify_id'],
                    set_={k: v for k, v in transformed.items() if k != 'shopify_id'}
                )
                
                session.execute(stmt)
                count += 1
        
        logger.info(f"Loaded {count} customers into the database")
        return count
    
    def _load_products(self, products_data: List[Dict]) -> int:
        """Load product data into the database.
        
        Args:
            products_data: List of product data from Shopify API
            
        Returns:
            Number of products processed
        """
        count = 0
        
        with db_session() as session:
            for product in products_data:
                transformed = self._transform_product(product)
                
                # Use upsert pattern (insert or update)
                stmt = insert(Product).values(**transformed)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['shopify_id'],
                    set_={k: v for k, v in transformed.items() if k != 'shopify_id'}
                )
                
                session.execute(stmt)
                count += 1
        
        logger.info(f"Loaded {count} products into the database")
        return count
    
    def _load_orders(self, orders_data: List[Dict]) -> int:
        """Load order data into the database.
        
        Args:
            orders_data: List of order data from Shopify API
            
        Returns:
            Number of orders processed
        """
        count = 0
        
        with db_session() as session:
            for order in orders_data:
                # First, ensure we have the customer record
                customer_id = None
                if order.get('customer'):
                    shopify_customer_id = str(order['customer'].get('id'))
                    customer = session.query(Customer).filter_by(shopify_id=shopify_customer_id).first()
                    
                    if not customer:
                        # Create a minimal customer record if it doesn't exist
                        customer_data = order['customer']
                        customer = Customer(
                            shopify_id=shopify_customer_id,
                            email=customer_data.get('email'),
                            first_name=customer_data.get('first_name'),
                            last_name=customer_data.get('last_name')
                        )
                        session.add(customer)
                        session.flush()  # To get the ID
                    
                    customer_id = customer.id
                
                if not customer_id:
                    # Skip orders without a customer
                    logger.warning(f"Skipping order {order.get('id')} with no customer")
                    continue
                
                transformed = self._transform_order(order)
                transformed['customer_id'] = customer_id
                
                # Use upsert pattern (insert or update)
                stmt = insert(Order).values(**transformed)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['shopify_id'],
                    set_={k: v for k, v in transformed.items() if k != 'shopify_id'}
                )
                
                session.execute(stmt)
                count += 1
        
        logger.info(f"Loaded {count} orders into the database")
        return count
    
    def extract_and_load(self, days_ago: int = 30, save_raw: bool = True) -> Dict[str, int]:
        """Extract data from Shopify and load it into the database.
        
        Args:
            days_ago: Number of days to look back for data
            save_raw: Whether to save raw data to files
            
        Returns:
            Dictionary with counts of processed records
        """
        logger.info(f"Starting Shopify ETL process, looking back {days_ago} days")
        
        # Extract data
        data = self.shopify_client.extract_daily_data(days_ago=days_ago)
        
        # Save raw data if requested
        if save_raw:
            self._save_raw_data(data)
        
        # Load data into database
        customers_count = self._load_customers(data.get('customers', []))
        products_count = self._load_products(data.get('products', []))
        orders_count = self._load_orders(data.get('orders', []))
        
        logger.info("Shopify ETL process completed successfully")
        
        return {
            'customers': customers_count,
            'products': products_count,
            'orders': orders_count
        }


def run_etl(days_ago: int = 30, save_raw: bool = True) -> Dict[str, int]:
    """Run the Shopify ETL process.
    
    Args:
        days_ago: Number of days to look back for data
        save_raw: Whether to save raw data to files
        
    Returns:
        Dictionary with counts of processed records
    """
    etl = ShopifyETL()
    return etl.extract_and_load(days_ago=days_ago, save_raw=save_raw)


if __name__ == "__main__":
    """Run the ETL process when executed directly."""
    run_etl() 