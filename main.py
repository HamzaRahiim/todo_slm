import os
import requests
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    PHI3_MODEL = os.getenv("PHI3_MODEL", "phi3:latest")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "30"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    PORT = int(os.getenv("PORT", "8000"))

config = Config()

# Global storage
todos = []
todo_counter = 1

# Pydantic Models (Fixed for Railway)
class Todo(BaseModel):
    id: Optional[int] = None
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    completed: bool = False
    created_at: Optional[str] = None
    priority: Optional[str] = "medium"  # Simplified without pattern validation

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    response: str
    model_used: str = "phi3"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time: Optional[float] = None
    fallback_used: bool = False

# Ollama Service with Fallbacks
class OllamaService:
    def __init__(self):
        self.url = config.OLLAMA_URL
        self.model = config.PHI3_MODEL
        self.is_healthy = False
        self.model_loaded = False
    
    async def check_health(self) -> bool:
        """Check if Ollama is healthy"""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                self.is_healthy = True
                self.model_loaded = any("phi3" in name.lower() for name in model_names)
                
                logger.info(f"✅ Ollama healthy: {len(models)} models, Phi-3: {self.model_loaded}")
                return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
        
        self.is_healthy = False
        self.model_loaded = False
        return False
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> tuple[str, float, bool]:
        """Generate response with fallback to mock responses"""
        start_time = time.time()
        
        # Try real Phi-3 first
        if self.is_healthy and self.model_loaded:
            try:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:" if system_prompt else prompt
                
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 2048,
                        "num_predict": 256,
                        "top_p": 0.9
                    }
                }
                
                response = requests.post(
                    f"{self.url}/api/generate",
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "").strip()
                    
                    if ai_response:
                        logger.info(f"✅ Phi-3 responded in {processing_time:.2f}s")
                        return ai_response, processing_time, False
                
            except Exception as e:
                logger.warning(f"Phi-3 request failed: {e}")
        
        # Fallback to intelligent mock responses
        processing_time = time.time() - start_time
        fallback_response = self._generate_fallback_response(prompt)
        logger.info(f"🔄 Using fallback response in {processing_time:.2f}s")
        return fallback_response, processing_time, True
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Intelligent fallback responses"""
        prompt_lower = prompt.lower()
        
        # Todo-related responses
        if any(word in prompt_lower for word in ["todo", "task", "item"]):
            if any(word in prompt_lower for word in ["add", "create", "new", "make"]):
                return "I'd be happy to help you create a new todo! You can add tasks using the /todos endpoint. What would you like to accomplish today?"
            elif any(word in prompt_lower for word in ["show", "list", "display", "see"]):
                return f"You currently have {len(todos)} todos in your list. {len([t for t in todos if not t.completed])} are still pending. You can view them all at the /todos endpoint!"
            elif any(word in prompt_lower for word in ["complete", "done", "finish"]):
                return "Great job on staying productive! You can mark todos as complete by updating them through the /todos/{id} endpoint. Keep up the excellent work!"
            elif any(word in prompt_lower for word in ["delete", "remove"]):
                return "If you need to remove a todo, you can delete it using the API. Sometimes it's good to clean up completed or outdated tasks!"
        
        # Productivity advice
        elif any(word in prompt_lower for word in ["help", "advice", "tip", "productivity"]):
            tips = [
                "Here are some productivity tips: 1) Break large tasks into smaller, manageable pieces, 2) Prioritize your most important tasks first, 3) Set realistic deadlines, 4) Take regular breaks to maintain focus!",
                "For better task management: Use the priority field to organize your todos, group similar tasks together, and celebrate when you complete items!",
                "Time management tip: Try the Pomodoro Technique - work for 25 minutes, then take a 5-minute break. It's great for maintaining focus!"
            ]
            import random
            return random.choice(tips)
        
        # Greetings
        elif any(word in prompt_lower for word in ["hello", "hi", "hey", "greet"]):
            return "Hello! I'm your AI todo assistant. I'm currently running in fallback mode, but I can still help you manage your tasks and provide productivity advice. How can I assist you today?"
        
        # Default intelligent response
        else:
            return f"I understand you're asking about: '{prompt[:100]}...' While I'm running in fallback mode right now, I'm still here to help you manage your todos effectively! You can create, update, and organize your tasks through this API. What would you like to work on?"

# Initialize service
ollama_service = OllamaService()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting AI Todo App on Railway...")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"Port: {config.PORT}")
    
    # Check Ollama health (don't block startup if it fails)
    await ollama_service.check_health()
    
    if ollama_service.is_healthy:
        logger.info("✅ Ollama is available")
        if ollama_service.model_loaded:
            logger.info("✅ Phi-3 model loaded successfully")
        else:
            logger.warning("⚠️ Phi-3 model not loaded, using fallbacks")
    else:
        logger.warning("⚠️ Ollama not available, using fallback responses")
    
    logger.info("✅ Application startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="AI Todo App with Phi-3",
    description="Railway-deployed todo app with AI assistance",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {
        "message": "🤖 AI Todo App with Phi-3",
        "status": "running",
        "environment": config.ENVIRONMENT,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "todos": "/todos",
            "chat": "/chat"
        }
    }

@app.get("/health")
async def health_check():
    """Health check with service status"""
    # Quick health check
    ollama_healthy = await ollama_service.check_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": config.ENVIRONMENT,
        "services": {
            "fastapi": "healthy",
            "ollama": "healthy" if ollama_healthy else "degraded",
            "phi3_model": "loaded" if ollama_service.model_loaded else "fallback"
        },
        "app": {
            "todos_count": len(todos),
            "port": config.PORT
        }
    }

@app.post("/todos", response_model=Todo)
async def create_todo(todo: Todo):
    """Create a new todo"""
    global todo_counter
    
    new_todo = Todo(
        id=todo_counter,
        title=todo.title.strip(),
        description=todo.description.strip() if todo.description else "",
        completed=False,
        created_at=datetime.now().isoformat(),
        priority=todo.priority or "medium"
    )
    
    todos.append(new_todo)
    todo_counter += 1
    
    logger.info(f"📝 Created todo: {new_todo.title}")
    return new_todo

@app.get("/todos", response_model=List[Todo])
async def get_todos():
    """Get all todos"""
    return todos

@app.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, updated_todo: Todo):
    """Update a todo"""
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[i].title = updated_todo.title.strip()
            todos[i].description = updated_todo.description.strip() if updated_todo.description else ""
            todos[i].completed = updated_todo.completed
            todos[i].priority = updated_todo.priority or "medium"
            
            logger.info(f"✏️ Updated todo {todo_id}")
            return todos[i]
    
    raise HTTPException(status_code=404, detail="Todo not found")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI (Phi-3 or fallback)"""
    
    # Enhanced system prompt
    system_prompt = f"""You are a helpful AI assistant for todo management. 

Current system state:
- Total todos: {len(todos)}
- Pending: {len([t for t in todos if not t.completed])}
- Completed: {len([t for t in todos if t.completed])}

Recent todos: {', '.join([t.title for t in todos[-3:]]) if todos else 'None yet'}

Be helpful, encouraging, and provide practical advice about task management."""

    # Generate response
    ai_response, processing_time, fallback_used = ollama_service.generate_response(
        request.message, 
        system_prompt
    )
    
    # Log the interaction
    logger.info(f"💬 Chat - Fallback: {fallback_used} | Time: {processing_time:.2f}s")
    
    return ChatResponse(
        response=ai_response,
        model_used="phi3" if not fallback_used else "phi3-fallback",
        processing_time=processing_time,
        fallback_used=fallback_used
    )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Something went wrong"}
    )

# Main execution
if __name__ == "__main__":
    logger.info(f"🚀 Starting server on port {config.PORT}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=False,
        log_level="info"
    )