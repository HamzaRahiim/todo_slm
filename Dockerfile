# Dockerfile (All-in-one: Ollama + FastAPI + Phi-3)
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pre-pull Phi-3 model at build time
RUN /root/.ollama/bin/ollama pull phi3:latest

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# Expose ports
EXPOSE 8000 11434

# Start everything under supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]