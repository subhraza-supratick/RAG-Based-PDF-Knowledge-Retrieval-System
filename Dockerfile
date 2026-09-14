FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/uploads data

# Expose Django port
EXPOSE 8000

# Environment variables
ENV DJANGO_SETTINGS_MODULE=document_rag.settings
ENV PYTHONUNBUFFERED=1

# Run Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
