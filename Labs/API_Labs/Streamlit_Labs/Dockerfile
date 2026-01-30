# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY data/ ./data/
COPY src/ ./src/

# Create model directory if it doesn't exist
RUN mkdir -p /app/model

# Expose port 8000 (default FastAPI port)
EXPOSE 8000

# Command to run the application
# Using uvicorn to run the FastAPI app
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 specifies the port (you can change this)
WORKDIR /app/src
CMD ["streamlit", "run", "Dashboard.py"]