FROM python:3.11-slim

# Install basics
RUN apt-get update && apt-get install -y curl supervisor && rm -rf /var/lib/apt/lists/*

# Copy your app
WORKDIR /app
COPY . /app

# Supervisor will keep Ollama + your app running
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
