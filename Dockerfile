# Railway-compatible Dockerfile (CPU-only, no CUDA)
FROM ubuntu:22.04

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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

# Install Ollama (CPU version)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create optimized startup script for Railway
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Railway deployment..."\n\
\n\
# Start Ollama service in background\n\
echo "📡 Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Function to cleanup on exit\n\
cleanup() {\n\
    echo "🛑 Shutting down services..."\n\
    kill $OLLAMA_PID 2>/dev/null || true\n\
    exit\n\
}\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Wait for Ollama to be ready with timeout\n\
echo "⏳ Waiting for Ollama to initialize..."\n\
timeout=120\n\
counter=0\n\
while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do\n\
    counter=$((counter + 1))\n\
    if [ $counter -gt $timeout ]; then\n\
        echo "❌ Ollama failed to start within $timeout seconds"\n\
        echo "🔄 Continuing with FastAPI only (Ollama will be marked as unhealthy)"\n\
        break\n\
    fi\n\
    echo "⏳ Waiting for Ollama... ($counter/$timeout)"\n\
    sleep 1\n\
done\n\
\n\
if [ $counter -le $timeout ]; then\n\
    echo "✅ Ollama is ready"\n\
    \n\
    # Download Phi-3 model with timeout\n\
    echo "📥 Downloading Phi-3 model..."\n\
    timeout 600 ollama pull phi3:latest || {\n\
        echo "⚠️ Model download failed or timed out, continuing with FastAPI"\n\
        echo "🔄 The app will use fallback responses until model is available"\n\
    }\n\
    \n\
    # Quick model test\n\
    echo "🧪 Testing model..."\n\
    timeout 30 ollama run phi3:latest "Hello" > /dev/null 2>&1 && {\n\
        echo "✅ Phi-3 model is working"\n\
    } || {\n\
        echo "⚠️ Model test failed, but continuing with FastAPI"\n\
    }\n\
else\n\
    echo "⚠️ Ollama not ready, but FastAPI will still start"\n\
fi\n\
\n\
# Start FastAPI application\n\
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."\n\
exec python3 main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:${PORT:-8000}/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Switch to non-root user
USER appuser

# Expose port (Railway will set PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD /app/healthcheck.sh

# Start the application
CMD ["/app/start.sh"]