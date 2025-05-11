"""Customer API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.database import db_session
from src.models.entities import Customer, Order, Review, FeatureStat
from src.models.customer_segmentation import CustomerSegmenter, run_segmentation
from src.features.customer_features import CustomerFeatureEngineer, run_feature_engineering

router = APIRouter()
logger = get_logger(__name__)
config = get_config()


def get_db():
    """Get database session."""
    with db_session() as session:
        yield session


@router.get("/")
async def get_customers(
    skip: int = Query(0, ge=0, description="Number of customers to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of customers to return"),
    db: Session = Depends(get_db)
):
    """Get a list of customers.
    
    Args:
        skip: Number of customers to skip (pagination)
        limit: Maximum number of customers to return
        db: Database session
        
    Returns:
        List of customers
    """
    customers = db.query(Customer).offset(skip).limit(limit).all()
    
    result = []
    for customer in customers:
        result.append({
            "id": str(customer.id),
            "shopify_id": customer.shopify_id,
            "email": customer.email,
            "first_name": customer.first_name,
            "last_name": customer.last_name,
            "created_at": customer.created_at.isoformat()
        })
    
    return {"customers": result, "total": len(result), "skip": skip, "limit": limit}


@router.get("/{customer_id}")
async def get_customer(
    customer_id: str = Path(..., description="Customer ID"),
    db: Session = Depends(get_db)
):
    """Get a customer by ID.
    
    Args:
        customer_id: Customer ID
        db: Database session
        
    Returns:
        Customer details
    """
    try:
        customer_uuid = uuid.UUID(customer_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid customer ID format")
    
    customer = db.query(Customer).filter(Customer.id == customer_uuid).first()
    
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Get customer orders
    orders = db.query(Order).filter(Order.customer_id == customer_uuid).all()
    
    # Get customer reviews
    reviews = db.query(Review).filter(Review.customer_id == customer_uuid).all()
    
    # Get customer features
    features = db.query(FeatureStat).filter(
        FeatureStat.entity_type == 'customer',
        FeatureStat.entity_id == customer_uuid
    ).all()
    
    # Format the response
    result = {
        "id": str(customer.id),
        "shopify_id": customer.shopify_id,
        "email": customer.email,
        "first_name": customer.first_name,
        "last_name": customer.last_name,
        "phone": customer.phone,
        "verified_email": customer.verified_email,
        "accepts_marketing": customer.accepts_marketing,
        "created_at": customer.created_at.isoformat(),
        "updated_at": customer.updated_at.isoformat(),
        "metadata": customer.metadata,
        "orders": [
            {
                "id": str(order.id),
                "order_number": order.order_number,
                "total_price": order.total_price,
                "created_at": order.created_at.isoformat()
            }
            for order in orders
        ],
        "reviews": [
            {
                "id": str(review.id),
                "product_id": str(review.product_id),
                "rating": review.rating,
                "title": review.title,
                "created_at": review.created_at.isoformat()
            }
            for review in reviews
        ],
        "features": {
            feature.feature_name: feature.value
            for feature in features
        }
    }
    
    return result


@router.get("/{customer_id}/orders")
async def get_customer_orders(
    customer_id: str = Path(..., description="Customer ID"),
    skip: int = Query(0, ge=0, description="Number of orders to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders to return"),
    db: Session = Depends(get_db)
):
    """Get orders for a customer.
    
    Args:
        customer_id: Customer ID
        skip: Number of orders to skip (pagination)
        limit: Maximum number of orders to return
        db: Database session
        
    Returns:
        List of customer orders
    """
    try:
        customer_uuid = uuid.UUID(customer_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid customer ID format")
    
    # Check if customer exists
    customer = db.query(Customer).filter(Customer.id == customer_uuid).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Get customer orders
    orders = db.query(Order).filter(Order.customer_id == customer_uuid)\
        .order_by(Order.created_at.desc())\
        .offset(skip).limit(limit).all()
    
    result = []
    for order in orders:
        result.append({
            "id": str(order.id),
            "shopify_id": order.shopify_id,
            "order_number": order.order_number,
            "total_price": order.total_price,
            "subtotal_price": order.subtotal_price,
            "total_tax": order.total_tax,
            "total_discounts": order.total_discounts,
            "total_shipping": order.total_shipping,
            "currency": order.currency,
            "financial_status": order.financial_status,
            "fulfillment_status": order.fulfillment_status,
            "created_at": order.created_at.isoformat()
        })
    
    return {"orders": result, "total": len(result), "customer_id": customer_id, "skip": skip, "limit": limit}


@router.get("/segmentation/segments")
async def get_customer_segments(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of customers to process"),
    n_clusters: int = Query(4, ge=2, le=10, description="Number of clusters for segmentation"),
    db: Session = Depends(get_db)
):
    """Run customer segmentation and return segments.
    
    Args:
        limit: Maximum number of customers to process
        n_clusters: Number of clusters for segmentation
        db: Database session
        
    Returns:
        Customer segmentation results
    """
    try:
        # Run segmentation
        segmenter, segments_df = run_segmentation(n_clusters=n_clusters, limit=limit, log_to_mlflow=True)
        
        if len(segments_df) == 0:
            return {"error": "No customers available for segmentation", "segments": []}
        
        # Convert segments to list of dictionaries
        segments = segments_df.to_dict(orient='records')
        
        # Get segment statistics
        segment_counts = segments_df['segment'].value_counts().to_dict()
        segment_stats = {}
        
        # Load features to get segment profiles
        features_df = segmenter.load_features(limit=limit)
        
        # Merge features with segments
        merged_df = pd.merge(features_df, segments_df, on='customer_id')
        
        # Calculate segment profiles
        if not merged_df.empty:
            segment_profiles = merged_df.groupby('segment')[segmenter.feature_columns].mean().to_dict(orient='index')
            
            for segment, profile in segment_profiles.items():
                segment_stats[segment] = {
                    "count": segment_counts.get(segment, 0),
                    "percentage": segment_counts.get(segment, 0) / len(segments_df) * 100,
                    "profile": profile
                }
        
        return {
            "total_customers": len(segments_df),
            "n_clusters": n_clusters,
            "segments": segments,
            "segment_stats": segment_stats
        }
    
    except Exception as e:
        logger.error(f"Error running customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running customer segmentation: {str(e)}")


@router.get("/segmentation/customer/{customer_id}")
async def get_customer_segment(
    customer_id: str = Path(..., description="Customer ID"),
    db: Session = Depends(get_db)
):
    """Get segment for a specific customer.
    
    Args:
        customer_id: Customer ID
        db: Database session
        
    Returns:
        Customer segment information
    """
    try:
        customer_uuid = uuid.UUID(customer_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid customer ID format")
    
    # Check if customer exists
    customer = db.query(Customer).filter(Customer.id == customer_uuid).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # First check if we have segment information in the database
    segment_feature = db.query(FeatureStat).filter(
        FeatureStat.entity_type == 'customer',
        FeatureStat.entity_id == customer_uuid,
        FeatureStat.feature_name == 'segment'
    ).first()
    
    if segment_feature:
        segment = int(segment_feature.value)
    else:
        # Need to run segmentation for this customer
        # Load customer features
        features = db.query(FeatureStat).filter(
            FeatureStat.entity_type == 'customer',
            FeatureStat.entity_id == customer_uuid
        ).all()
        
        if not features:
            # Need to compute features first
            engineer = CustomerFeatureEngineer()
            customer_df = pd.DataFrame([{
                'customer_id': customer_uuid,
                'email': customer.email,
                'first_name': customer.first_name,
                'last_name': customer.last_name,
                'created_at': customer.created_at,
                'updated_at': customer.updated_at
            }])
            features_df = engineer.compute_all_features(limit=1)
            engineer.save_features_to_db(features_df)
        
        # Now run segmentation
        segmenter = CustomerSegmenter()
        
        # Create a dataframe with this customer's features
        features_data = []
        for feature in db.query(FeatureStat).filter(
            FeatureStat.entity_type == 'customer',
            FeatureStat.entity_id == customer_uuid
        ).all():
            features_data.append({
                'customer_id': customer_uuid,
                'feature_name': feature.feature_name,
                'value': feature.value
            })
        
        if not features_data:
            raise HTTPException(status_code=404, detail="Customer features not found")
        
        df = pd.DataFrame(features_data)
        pivoted = df.pivot(index='customer_id', columns='feature_name', values='value').reset_index()
        
        # Predict segment
        try:
            segments_df = segmenter.predict(pivoted)
            segment = int(segments_df.iloc[0]['segment'])
            
            # Save segment to database
            db.add(FeatureStat(
                entity_type='customer',
                entity_id=customer_uuid,
                feature_name='segment',
                stat_type='value',
                value=float(segment),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ))
            db.commit()
        except Exception as e:
            logger.error(f"Error predicting segment: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error predicting segment: {str(e)}")
    
    # Get human-readable segment description
    segment_descriptions = {
        0: "High Value (high monetary value, frequent purchases)",
        1: "Loyal Customers (frequent purchases, moderate value)",
        2: "New Customers (recent first purchase)",
        3: "At Risk (high value, not recent purchases)"
    }
    
    segment_description = segment_descriptions.get(segment, f"Segment {segment}")
    
    return {
        "customer_id": customer_id,
        "segment": segment,
        "segment_description": segment_description
    }


@router.post("/features/generate")
async def generate_customer_features(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of customers to process"),
    db: Session = Depends(get_db)
):
    """Generate features for customers.
    
    Args:
        limit: Maximum number of customers to process
        db: Database session
        
    Returns:
        Feature generation results
    """
    try:
        # Run feature engineering
        features_df = run_feature_engineering(limit=limit, save_to_db=True, log_to_mlflow=True)
        
        return {
            "success": True,
            "total_customers": len(features_df),
            "features_generated": len(features_df.columns) - 1  # Subtract customer_id column
        }
    
    except Exception as e:
        logger.error(f"Error generating customer features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating customer features: {str(e)}")


@router.get("/features/{customer_id}")
async def get_customer_features(
    customer_id: str = Path(..., description="Customer ID"),
    db: Session = Depends(get_db)
):
    """Get features for a specific customer.
    
    Args:
        customer_id: Customer ID
        db: Database session
        
    Returns:
        Customer features
    """
    try:
        customer_uuid = uuid.UUID(customer_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid customer ID format")
    
    # Check if customer exists
    customer = db.query(Customer).filter(Customer.id == customer_uuid).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Get customer features
    features = db.query(FeatureStat).filter(
        FeatureStat.entity_type == 'customer',
        FeatureStat.entity_id == customer_uuid
    ).all()
    
    if not features:
        # Need to compute features for this customer
        try:
            engineer = CustomerFeatureEngineer()
            customer_df = pd.DataFrame([{
                'customer_id': customer_uuid,
                'email': customer.email,
                'first_name': customer.first_name,
                'last_name': customer.last_name,
                'created_at': customer.created_at,
                'updated_at': customer.updated_at
            }])
            
            # Load orders for this customer
            orders_df = engineer.load_order_data(customer_ids=[customer_uuid])
            
            # Compute features
            if len(orders_df) > 0:
                rfm_df = engineer.compute_recency_frequency_monetary(orders_df)
                purchase_patterns_df = engineer.compute_purchase_patterns(orders_df)
                clv_df = engineer.compute_customer_lifetime_value(rfm_df, purchase_patterns_df)
                
                # Merge all features
                features_df = customer_df.merge(rfm_df, on='customer_id', how='left')
                features_df = features_df.merge(purchase_patterns_df, on='customer_id', how='left')
                features_df = features_df.merge(clv_df, on='customer_id', how='left')
                
                # Fill NaN values
                features_df = features_df.fillna({
                    'recency_days': -1,
                    'frequency': 0,
                    'monetary': 0,
                    'avg_order_value': 0,
                    'std_order_value': 0,
                    'days_between_orders': -1,
                    'customer_lifetime_value': 0
                })
                
                # Save features to database
                engineer.save_features_to_db(features_df)
            else:
                # No orders, set default features
                features_df = customer_df.assign(
                    recency_days=-1,
                    frequency=0,
                    monetary=0,
                    avg_order_value=0,
                    std_order_value=0,
                    days_between_orders=-1,
                    customer_lifetime_value=0
                )
                engineer.save_features_to_db(features_df)
            
            # Reload features from database
            features = db.query(FeatureStat).filter(
                FeatureStat.entity_type == 'customer',
                FeatureStat.entity_id == customer_uuid
            ).all()
        
        except Exception as e:
            logger.error(f"Error computing features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error computing features: {str(e)}")
    
    if not features:
        raise HTTPException(status_code=404, detail="Failed to generate customer features")
    
    # Format the response
    result = {
        "customer_id": customer_id,
        "features": {
            feature.feature_name: feature.value
            for feature in features
        },
        "updated_at": max(feature.updated_at for feature in features).isoformat()
    }
    
    return result


@router.get("/stats/summary")
async def get_customer_stats(
    db: Session = Depends(get_db)
):
    """Get summary statistics for customers.
    
    Args:
        db: Database session
        
    Returns:
        Customer statistics
    """
    # Total customers
    total_customers = db.query(func.count(Customer.id)).scalar()
    
    # New customers in last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    new_customers = db.query(func.count(Customer.id)).filter(Customer.created_at >= thirty_days_ago).scalar()
    
    # Customers with orders
    customers_with_orders = db.query(func.count(func.distinct(Order.customer_id))).scalar()
    
    # Average orders per customer
    avg_orders = db.query(func.count(Order.id) / func.count(func.distinct(Order.customer_id))).scalar()
    
    # Average order value
    avg_order_value = db.query(func.avg(Order.total_price)).scalar()
    
    return {
        "total_customers": total_customers,
        "new_customers_30d": new_customers,
        "customers_with_orders": customers_with_orders,
        "customers_without_orders": total_customers - customers_with_orders,
        "average_orders_per_customer": float(avg_orders) if avg_orders else 0,
        "average_order_value": float(avg_order_value) if avg_order_value else 0
    } 