"""Shopify API client for the E-commerce Analytics Platform."""

import time
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class ShopifyAPIClient:
    """Client for interacting with the Shopify API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 shop_name: Optional[str] = None, api_version: Optional[str] = None):
        """Initialize the Shopify API client.
        
        Args:
            api_key: Shopify API key. If None, read from config
            api_secret: Shopify API secret. If None, read from config
            shop_name: Shopify shop name. If None, read from config
            api_version: Shopify API version. If None, read from config
        """
        self.api_key = api_key or config.get('shopify.api_key')
        self.api_secret = api_secret or config.get('shopify.api_secret')
        self.shop_name = shop_name or config.get('shopify.shop_name')
        self.api_version = api_version or config.get('shopify.api_version', '2023-07')
        
        if not self.api_key or not self.api_secret or not self.shop_name:
            raise ValueError("Shopify API credentials are required")
        
        self.base_url = f"https://{self.shop_name}.myshopify.com/admin/api/{self.api_version}"
        self.session = requests.Session()
        self.session.auth = (self.api_key, self.api_secret)
        self.session.headers.update({'Content-Type': 'application/json'})
        
        logger.info(f"Initialized Shopify API client for shop: {self.shop_name}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                      data: Optional[Dict] = None, retries: int = 3) -> Dict:
        """Make a request to the Shopify API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            retries: Number of retry attempts for rate limiting
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params)
                elif method == 'POST':
                    response = self.session.post(url, params=params, json=data)
                elif method == 'PUT':
                    response = self.session.put(url, params=params, json=data)
                elif method == 'DELETE':
                    response = self.session.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                # Check if we hit rate limits
                if response.status_code == 429 and attempt < retries - 1:
                    # Get retry-after header or use exponential backoff
                    retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                    logger.warning(f"Rate limited by Shopify API. Retrying in {retry_after} seconds")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Shopify API request failed: {str(e)}")
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Shopify API request failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def get_products(self, limit: int = 50, page_info: Optional[str] = None, 
                     created_at_min: Optional[str] = None, updated_at_min: Optional[str] = None,
                     collection_id: Optional[str] = None) -> Dict:
        """Get products from Shopify.
        
        Args:
            limit: Number of results per page (max 250)
            page_info: Page info for pagination
            created_at_min: Filter by minimum creation date (ISO 8601 format)
            updated_at_min: Filter by minimum update date (ISO 8601 format)
            collection_id: Filter by collection ID
            
        Returns:
            Dictionary containing products and pagination info
        """
        params = {'limit': min(limit, 250)}
        
        if page_info:
            params['page_info'] = page_info
        
        if created_at_min:
            params['created_at_min'] = created_at_min
            
        if updated_at_min:
            params['updated_at_min'] = updated_at_min
            
        if collection_id:
            params['collection_id'] = collection_id
        
        return self._make_request('GET', 'products.json', params=params)
    
    def get_orders(self, limit: int = 50, page_info: Optional[str] = None,
                   status: str = 'any', financial_status: Optional[str] = None,
                   fulfillment_status: Optional[str] = None, created_at_min: Optional[str] = None,
                   updated_at_min: Optional[str] = None) -> Dict:
        """Get orders from Shopify.
        
        Args:
            limit: Number of results per page (max 250)
            page_info: Page info for pagination
            status: Order status (open, closed, cancelled, any)
            financial_status: Payment status (authorized, paid, partially_paid, etc.)
            fulfillment_status: Shipping status (fulfilled, partially_fulfilled, etc.)
            created_at_min: Filter by minimum creation date (ISO 8601 format)
            updated_at_min: Filter by minimum update date (ISO 8601 format)
            
        Returns:
            Dictionary containing orders and pagination info
        """
        params = {
            'limit': min(limit, 250),
            'status': status
        }
        
        if page_info:
            params['page_info'] = page_info
            
        if financial_status:
            params['financial_status'] = financial_status
            
        if fulfillment_status:
            params['fulfillment_status'] = fulfillment_status
            
        if created_at_min:
            params['created_at_min'] = created_at_min
            
        if updated_at_min:
            params['updated_at_min'] = updated_at_min
        
        return self._make_request('GET', 'orders.json', params=params)
    
    def get_customers(self, limit: int = 50, page_info: Optional[str] = None,
                      created_at_min: Optional[str] = None, updated_at_min: Optional[str] = None) -> Dict:
        """Get customers from Shopify.
        
        Args:
            limit: Number of results per page (max 250)
            page_info: Page info for pagination
            created_at_min: Filter by minimum creation date (ISO 8601 format)
            updated_at_min: Filter by minimum update date (ISO 8601 format)
            
        Returns:
            Dictionary containing customers and pagination info
        """
        params = {'limit': min(limit, 250)}
        
        if page_info:
            params['page_info'] = page_info
            
        if created_at_min:
            params['created_at_min'] = created_at_min
            
        if updated_at_min:
            params['updated_at_min'] = updated_at_min
        
        return self._make_request('GET', 'customers.json', params=params)
    
    def get_product(self, product_id: str) -> Dict:
        """Get a single product by ID.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            Product data
        """
        return self._make_request('GET', f'products/{product_id}.json')
    
    def get_order(self, order_id: str) -> Dict:
        """Get a single order by ID.
        
        Args:
            order_id: Shopify order ID
            
        Returns:
            Order data
        """
        return self._make_request('GET', f'orders/{order_id}.json')
    
    def get_customer(self, customer_id: str) -> Dict:
        """Get a single customer by ID.
        
        Args:
            customer_id: Shopify customer ID
            
        Returns:
            Customer data
        """
        return self._make_request('GET', f'customers/{customer_id}.json')
    
    def paginate_all(self, resource_method, **kwargs) -> List[Dict]:
        """Paginate through all pages of a resource.
        
        Args:
            resource_method: Method to call for getting resources
            **kwargs: Arguments to pass to the resource method
            
        Returns:
            List of all resources
        """
        all_resources = []
        page_info = None
        
        while True:
            if page_info:
                kwargs['page_info'] = page_info
                
            response = resource_method(**kwargs)
            
            # Extract the resource key (products, orders, customers)
            resource_key = list(response.keys())[0]
            resources = response[resource_key]
            
            all_resources.extend(resources)
            
            # Check if there are more pages
            link_header = response.get('links', {}).get('next')
            if not link_header:
                break
                
            # Extract page_info from link header
            page_info = link_header.split('page_info=')[1].split('&')[0]
            
            # Rate limiting
            time.sleep(0.5)
        
        return all_resources
    
    def extract_all_products(self, **kwargs) -> List[Dict]:
        """Extract all products with pagination.
        
        Args:
            **kwargs: Arguments to pass to get_products
            
        Returns:
            List of all products
        """
        return self.paginate_all(self.get_products, **kwargs)
    
    def extract_all_orders(self, **kwargs) -> List[Dict]:
        """Extract all orders with pagination.
        
        Args:
            **kwargs: Arguments to pass to get_orders
            
        Returns:
            List of all orders
        """
        return self.paginate_all(self.get_orders, **kwargs)
    
    def extract_all_customers(self, **kwargs) -> List[Dict]:
        """Extract all customers with pagination.
        
        Args:
            **kwargs: Arguments to pass to get_customers
            
        Returns:
            List of all customers
        """
        return self.paginate_all(self.get_customers, **kwargs)
    
    def extract_daily_data(self, days_ago: int = 1) -> Dict[str, List[Dict]]:
        """Extract data updated in the last N days.
        
        Args:
            days_ago: Number of days to look back
            
        Returns:
            Dictionary containing products, orders, and customers data
        """
        updated_since = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        
        products = self.extract_all_products(updated_at_min=updated_since)
        orders = self.extract_all_orders(updated_at_min=updated_since)
        customers = self.extract_all_customers(updated_at_min=updated_since)
        
        return {
            'products': products,
            'orders': orders,
            'customers': customers
        } 