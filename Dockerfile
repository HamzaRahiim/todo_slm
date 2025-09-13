FROM python:3.11-slim

# Install curl + supervisor
RUN apt-get update && apt-get install -y curl supervisor && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependencies first (better caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Add supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port
EXPOSE 8000

# Run supervisor (starts Ollama + FastAPI)
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
