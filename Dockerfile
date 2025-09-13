# Dockerfile (All-in-one: Ollama + FastAPI + Phi-3)
FROM python:3.11-slim

# Install curl and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Ollama in background..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
echo "⏳ Waiting for Ollama to start..."\n\
sleep 10\n\
\n\
echo "📥 Downloading Phi-3 model..."\n\
ollama pull phi3:latest\n\
\n\
echo "🧠 Testing Phi-3..."\n\
ollama run phi3 "Say hello if you are working" --verbose\n\
\n\
echo "🌐 Starting FastAPI server..."\n\
python main.py\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 11434

# Start everything
CMD ["/app/start.sh"]