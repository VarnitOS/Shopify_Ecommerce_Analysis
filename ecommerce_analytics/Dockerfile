FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional NLP models
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy the rest of the application
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for different services
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501
# Flask/MLflow
EXPOSE 5000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 