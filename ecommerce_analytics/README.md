# E-commerce Analytics Platform

A comprehensive data science and machine learning platform for e-commerce analytics, leveraging Shopify API data.

## Features

- **Customer Analytics**: Segmentation, lifetime value prediction, churn prediction
- **Product Analytics**: Recommendation engine, product categorization, pricing optimization
- **Order Analytics**: Demand forecasting, sales prediction, anomaly detection
- **Review Analytics**: Sentiment analysis, topic modeling, review summarization
- **Interactive Dashboards**: Real-time KPI tracking and visualization

## Tech Stack

### Data Engineering
- Apache Airflow (orchestration)
- Kafka (streaming)
- Delta Lake (storage)
- Feast (feature store)

### Analytics & ML
- PyTorch/TensorFlow (deep learning)
- Scikit-learn (machine learning)
- Prophet (time series)
- Transformers (NLP)
- Pandas/NumPy (data processing)

### MLOps
- MLflow (experiment tracking)
- Docker/Kubernetes (containerization)
- FastAPI (model serving)
- Prometheus/Grafana (monitoring)

### Visualization
- Streamlit (dashboards)
- Plotly/Dash (interactive charts)

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Shopify API credentials

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ecommerce-analytics.git
cd ecommerce-analytics
```

2. Create a `.env` file with your credentials
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Start the services with Docker Compose
```bash
docker-compose up -d
```

4. Access the services:
   - API: http://localhost:8000/docs
   - Dashboard: http://localhost:8501
   - MLflow: http://localhost:5000
   - Airflow: http://localhost:8080

## Project Structure

```
ecommerce_analytics/
├── config/             # Configuration files
├── data/               # Data storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── api/            # FastAPI service
│   ├── etl/            # Data pipelines
│   ├── features/       # Feature engineering
│   ├── models/         # Model definitions
│   ├── visualization/  # Dashboards
│   └── utils/          # Utilities
├── tests/              # Test suite
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Service orchestration
└── requirements.txt    # Python dependencies
```

## Development

### Local Development Setup

1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run specific components
```bash
# Run API server
python -m src.api.main

# Run Streamlit dashboard
streamlit run src/visualization/dashboard.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 