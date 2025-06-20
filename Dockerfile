# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Update the apt-get sources to include contrib and non-free repositories
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list

# Install system dependencies for PyAudio and other libraries
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libportaudio2-dev \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/audio

# Expose the port that Flask will run on
EXPOSE 5000

# Set environment variables (optional, can be overridden in docker-compose.yml)
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]