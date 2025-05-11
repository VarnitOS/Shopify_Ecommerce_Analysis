"""Airflow DAG for the Shopify ETL process."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.models import Variable
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.etl.shopify_etl import run_etl
from src.utils.logger import get_logger

logger = get_logger("shopify_etl_dag")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'shopify_etl_dag',
    default_args=default_args,
    description='Extract, transform, and load data from Shopify API',
    schedule_interval='0 1 * * *',  # Run daily at 1:00 AM
    catchup=False,
    max_active_runs=1,
    tags=['shopify', 'etl', 'ecommerce'],
)

# Define the tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

# Task to extract and load customer data
def extract_and_load_data(**kwargs):
    """Extract and load Shopify data."""
    execution_date = kwargs['execution_date']
    logger.info(f"Running Shopify ETL for execution date: {execution_date}")
    
    # Get days to look back from Airflow variables, or use default
    days_ago = Variable.get('shopify_etl_days_ago', default_var=1)
    try:
        days_ago = int(days_ago)
    except ValueError:
        days_ago = 1
    
    # Run the ETL process
    result = run_etl(days_ago=days_ago, save_raw=True)
    
    # Log the results
    logger.info(f"Shopify ETL results: {result}")
    
    return result

etl_task = PythonOperator(
    task_id='extract_and_load_data',
    python_callable=extract_and_load_data,
    provide_context=True,
    dag=dag,
)

# Task for data quality checks
def data_quality_checks(**kwargs):
    """Perform data quality checks on loaded data."""
    ti = kwargs['ti']
    etl_result = ti.xcom_pull(task_ids='extract_and_load_data')
    
    logger.info(f"Starting data quality checks for ETL result: {etl_result}")
    
    # Example data quality checks could be implemented here
    # For now, just check if we got some data
    if not etl_result:
        raise ValueError("ETL process returned no results")
    
    if sum(etl_result.values()) == 0:
        raise ValueError("No records were processed during ETL")
    
    logger.info("Data quality checks passed successfully")
    return True

data_quality_task = PythonOperator(
    task_id='data_quality_checks',
    python_callable=data_quality_checks,
    provide_context=True,
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start_task >> etl_task >> data_quality_task >> end_task 