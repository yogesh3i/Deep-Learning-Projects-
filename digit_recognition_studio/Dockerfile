# Use official, lightweight Python base image
FROM python:3.10-slim

# Set environment variable to avoid writing python bytecode and buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set working directory inside container
WORKDIR /code

# Install system utilities needed for building packages or scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the entire project directory contents to the container working directory
COPY . /code

# Expose port 7860 (Hugging Face default bind port)
EXPOSE 7860

# Command to execute Flask application
CMD ["python", "app.py"]
