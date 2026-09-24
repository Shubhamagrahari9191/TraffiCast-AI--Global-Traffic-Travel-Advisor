# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies (for building any C-extension packages if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose standard web port
EXPOSE 5000

# Run with Gunicorn WSGI production server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} server:app --workers 2 --timeout 120"]
