FROM ollama/ollama:latest

# Install Python & Supervisor
RUN apt-get update && apt-get install -y python3 python3-pip supervisor && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY supervisord.conf /etc/supervisor/conf.d/app.conf

EXPOSE 8000 11434

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
