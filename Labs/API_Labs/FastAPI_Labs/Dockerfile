# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY model/ ./model/
COPY src/ ./src/

# Create model directory if it doesn't exist
RUN mkdir -p /app/model

# Expose port 8000 (default FastAPI port)
EXPOSE 8000

# Command to run the application
# Using uvicorn to run the FastAPI app

WORKDIR /app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]