# Dockerfile for NYC Taxi Tip Prediction API
# Uses a slim Python base image to keep the final image small.

# 1. Base image
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (layer caching optimisation)
COPY requirements.txt .

# 4. Install Python dependencies (no pip cache → smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and model artifacts
COPY app.py .
COPY models/ ./models/

# 6. Expose the port uvicorn will listen on
EXPOSE 8000

# 7. Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
