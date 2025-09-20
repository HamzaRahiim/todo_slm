# Production-ready Dockerfile with multi-stage build
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create app user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /var/log/supervisor && \
    chown -R appuser:appuser /app

# Python dependencies stage
FROM base as python-deps

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Application stage
FROM base as app

# Copy Python packages from previous stage
COPY --from=python-deps /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app
COPY --chown=appuser:appuser . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Configure Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8000/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Create optimized startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting production deployment..."\n\
\n\
# Start Ollama service\n\
echo "📡 Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready\n\
echo "⏳ Waiting for Ollama to initialize..."\n\
timeout=60\n\
counter=0\n\
while ! curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    counter=$((counter + 1))\n\
    if [ $counter -gt $timeout ]; then\n\
        echo "❌ Ollama failed to start within $timeout seconds"\n\
        exit 1\n\
    fi\n\
    echo "⏳ Waiting... ($counter/$timeout)"\n\
    sleep 1\n\
done\n\
\n\
echo "✅ Ollama is ready"\n\
\n\
# Check if model exists, if not download it\n\
echo "🔍 Checking for Phi-3 model..."\n\
if ! ollama list | grep -q "phi3"; then\n\
    echo "📥 Downloading Phi-3 model (this may take a while)..."\n\
    ollama pull phi3:latest || {\n\
        echo "❌ Failed to download Phi-3 model"\n\
        exit 1\n\
    }\n\
else\n\
    echo "✅ Phi-3 model already available"\n\
fi\n\
\n\
# Test the model\n\
echo "🧪 Testing Phi-3 model..."\n\
if echo "Hello" | ollama run phi3:latest > /dev/null 2>&1; then\n\
    echo "✅ Phi-3 model is working correctly"\n\
else\n\
    echo "❌ Phi-3 model test failed"\n\
    exit 1\n\
fi\n\
\n\
# Start FastAPI application\n\
echo "🌐 Starting FastAPI server..."\n\
exec python3 main.py\n\
' > /app/start-production.sh && chmod +x /app/start-production.sh

# Switch to non-root user
USER appuser

EXPOSE 8000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh

# Start the application
CMD ["/app/start-production.sh"]