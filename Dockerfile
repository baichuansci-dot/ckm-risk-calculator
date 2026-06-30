# Use Python 3.11 full image for maximum compatibility
FROM python:3.11

# Set working directory
WORKDIR /app

# Install additional system dependencies for image processing
# Full image already has build tools (gcc, g++, make, git, etc.)
RUN apt-get update && apt-get install -y \
    libtiff6 \
    libtiff-dev \
    libjpeg62-turbo \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Railway will set $PORT environment variable)
EXPOSE 8000

# Start command - use config file to handle PORT
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
