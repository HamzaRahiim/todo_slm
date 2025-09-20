import os
import requests
import time
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log') if os.path.exists('/app/logs') else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration with environment variables
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    PHI3_MODEL = os.getenv("PHI3_MODEL", "phi3:latest")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "30"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

config = Config()

# Global storage (use proper database in production)
todos = []
todo_counter = 1

# Pydantic Models with validation
class Todo(BaseModel):
    id: Optional[int] = None
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    completed: bool = False
    created_at: Optional[str] = None
    priority: Optional[str] = Field("medium", regex="^(low|medium|high)$")
    tags: Optional[List[str]] = Field(default_factory=list)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str = "phi3"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action_performed: Optional[str] = None
    todos_affected: Optional[List[Todo]] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    services: dict
    app: dict

# Ollama Service Class
class OllamaService:
    def __init__(self):
        self.url = config.OLLAMA_URL
        self.model = config.PHI3_MODEL
        self.is_healthy = False
        self.model_loaded = False
    
    async def wait_for_service(self, max_retries: int = None) -> bool:
        """Wait for Ollama service to be ready"""
        max_retries = max_retries or config.MAX_RETRIES
        logger.info(f"Checking Ollama connection at {self.url}...")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.url}/api/tags", timeout=5)
                if response.status_code == 200:
                    self.is_healthy = True
                    logger.info(f"✅ Ollama is ready at {self.url}")
                    return True
            except requests.exceptions.ConnectionError:
                logger.warning(f"⏳ Waiting for Ollama... ({i+1}/{max_retries})")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(2)
        
        logger.error("❌ Failed to connect to Ollama after maximum retries")
        return False
    
    def check_model_availability(self) -> bool:
        """Check if Phi-3 model is available"""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                phi3_available = any("phi3" in name.lower() for name in model_names)
                
                if phi3_available:
                    self.model_loaded = True
                    logger.info("✅ Phi-3 model is available")
                    return True
                else:
                    logger.error(f"❌ Phi-3 model not found. Available: {model_names}")
                    return False
        except Exception as e:
            logger.error(f"Error checking Phi-3 model: {e}")
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> tuple[str, float]:
        """Generate response using Phi-3 model"""
        start_time = time.time()
        
        try:
            # Construct the full prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 4096,
                    "num_predict": 512,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info("🤖 Sending request to Phi-3...")
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
                    return ai_response, processing_time
                else:
                    return "I apologize, but I couldn't generate a response. Please try again.", processing_time
            
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}", processing_time
                
        except requests.exceptions.Timeout:
            processing_time = time.time() - start_time
            error_msg = "Request timed out. The model might be processing a complex request."
            logger.warning(error_msg)
            return error_msg, processing_time
            
        except requests.exceptions.ConnectionError:
            processing_time = time.time() - start_time
            error_msg = "Cannot connect to Ollama service."
            logger.error(error_msg)
            return error_msg, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return error_msg, processing_time

# Initialize Ollama service
ollama_service = OllamaService()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting AI Todo App with Phi-3...")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"Ollama URL: {config.OLLAMA_URL}")
    logger.info(f"Model: {config.PHI3_MODEL}")
    
    # Wait for Ollama and check model
    if await ollama_service.wait_for_service():
        if ollama_service.check_model_availability():
            # Test the model
            logger.info("🧪 Testing Phi-3 model...")
            test_response, _ = ollama_service.generate_response(
                "Hello! Please confirm you are Phi-3 and working correctly."
            )
            logger.info(f"🤖 Test response: {test_response[:100]}...")
            logger.info("✅ Application startup complete!")
        else:
            logger.error("❌ Phi-3 model is not available!")
    else:
        logger.error("❌ Ollama service is not responding!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Professional AI Todo App with Phi-3",
    description="Production-ready todo app powered by Phi-3 model via Ollama",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware with more restrictive settings in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.ENVIRONMENT == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Todo CRUD Operations with improved error handling
@app.post("/todos", response_model=Todo)
async def create_todo(todo: Todo):
    """Create a new todo with validation"""
    global todo_counter
    
    try:
        new_todo = Todo(
            id=todo_counter,
            title=todo.title.strip(),
            description=todo.description.strip() if todo.description else "",
            completed=False,
            created_at=datetime.now().isoformat(),
            priority=todo.priority or "medium",
            tags=todo.tags or []
        )
        
        todos.append(new_todo)
        todo_counter += 1
        
        logger.info(f"📝 Created todo: {new_todo.title}")
        return new_todo
    except Exception as e:
        logger.error(f"Error creating todo: {e}")
        raise HTTPException(status_code=400, detail="Failed to create todo")


@app.get("/todos", response_model=List[Todo])
async def get_todos(
    completed: Optional[bool] = None,
    priority: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get todos with filtering and pagination"""
    filtered_todos = todos
    
    if completed is not None:
        filtered_todos = [t for t in filtered_todos if t.completed == completed]
    
    if priority:
        filtered_todos = [t for t in filtered_todos if t.priority == priority]
    
    return filtered_todos[offset:offset + limit]


@app.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int):
    """Get a specific todo by ID"""
    todo = next((t for t in todos if t.id == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo


@app.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, updated_todo: Todo):
    """Update a todo with validation"""
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[i].title = updated_todo.title.strip()
            todos[i].description = updated_todo.description.strip() if updated_todo.description else ""
            todos[i].completed = updated_todo.completed
            todos[i].priority = updated_todo.priority or "medium"
            todos[i].tags = updated_todo.tags or []
            
            logger.info(f"✏️ Updated todo {todo_id}: {updated_todo.title}")
            return todos[i]
    raise HTTPException(status_code=404, detail="Todo not found")


@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int):
    """Delete a todo"""
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            deleted_todo = todos.pop(i)
            logger.info(f"🗑️ Deleted todo {todo_id}: {deleted_todo.title}")
            return {"message": "Todo deleted successfully"}
    
    raise HTTPException(status_code=404, detail=f"Todo with ID {todo_id} not found")

# AI Chat Endpoint with improved system prompts
@app.post("/chat", response_model=ChatResponse)
async def chat_with_phi3(request: ChatRequest, background_tasks: BackgroundTasks):
    """Enhanced chat with Phi-3 for todo management"""
    
    if not ollama_service.is_healthy or not ollama_service.model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Phi-3 service is not available. Please check system health."
        )
    
    # Enhanced system prompt
    system_prompt = f"""You are an intelligent AI assistant for a professional todo application. You help users manage their tasks efficiently and provide productivity insights.

Current system state:
- Total todos: {len(todos)}
- Pending tasks: {len([t for t in todos if not t.completed])}
- Completed tasks: {len([t for t in todos if t.completed])}

Recent todos (last 5):
{chr(10).join([f"- ID {todo.id}: {todo.title} ({'✅ Done' if todo.completed else '⏳ Pending'}) [Priority: {todo.priority}]" for todo in todos[-5:]]) if todos else "No todos yet"}

Your capabilities:
1. Natural conversation about task management
2. Productivity advice and time management tips
3. Priority assessment and task organization suggestions
4. Encouraging and supportive responses
5. Context-aware task recommendations

Context from user: {request.context or 'No additional context provided'}

Respond naturally, be helpful, and maintain a professional yet friendly tone."""

    # Generate AI response
    ai_response, processing_time = ollama_service.generate_response(
        request.message, 
        system_prompt
    )
    
    # Enhanced intent detection
    message_lower = request.message.lower()
    action_performed = None
    todos_affected = []
    
    if any(word in message_lower for word in ["add", "create", "new", "make"]) and any(word in message_lower for word in ["todo", "task", "item"]):
        action_performed = "create_intent_detected"
    elif any(word in message_lower for word in ["show", "list", "display", "see", "view"]):
        action_performed = "list_intent_detected"
        todos_affected = todos.copy()
    elif any(word in message_lower for word in ["complete", "done", "finish", "mark"]):
        action_performed = "complete_intent_detected"
    elif any(word in message_lower for word in ["delete", "remove", "cancel"]):
        action_performed = "delete_intent_detected"
    elif any(word in message_lower for word in ["priority", "urgent", "important"]):
        action_performed = "priority_intent_detected"
    
    # Log the interaction
    logger.info(f"💬 Chat - User: {request.message[:50]}... | Response time: {processing_time:.2f}s")
    
    return ChatResponse(
        response=ai_response,
        model_used=config.PHI3_MODEL,
        action_performed=action_performed,
        todos_affected=todos_affected,
        processing_time=processing_time
    )

# Enhanced health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with detailed status"""
    
    # Check Ollama connection
    ollama_status = {"status": "unknown", "error": None, "response_time": None}
    
    start_time = time.time()
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            ollama_status = {
                "status": "healthy",
                "response_time": f"{response_time:.2f}s",
                "models_count": len(response.json().get("models", []))
            }
        else:
            ollama_status = {
                "status": f"error_{response.status_code}",
                "response_time": f"{response_time:.2f}s",
                "error": response.text
            }
    except Exception as e:
        ollama_status = {
            "status": "unhealthy",
            "error": str(e),
            "response_time": f"{time.time() - start_time:.2f}s"
        }
    
    # Check Phi-3 model
    phi3_status = {"status": "unknown", "test_response": None}
    
    if ollama_status["status"] == "healthy":
        if ollama_service.check_model_availability():
            test_response, test_time = ollama_service.generate_response(
                "Respond with 'OK' if you are working correctly."
            )
            phi3_status = {
                "status": "healthy" if "ok" in test_response.lower() else "responding_but_unclear",
                "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
                "response_time": f"{test_time:.2f}s"
            }
        else:
            phi3_status = {"status": "model_not_found"}
    else:
        phi3_status = {"status": "ollama_unavailable"}
    
    overall_status = "healthy" if (
        ollama_status["status"] == "healthy" and 
        phi3_status["status"] == "healthy"
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        environment=config.ENVIRONMENT,
        services={
            "ollama": {
                "url": config.OLLAMA_URL,
                **ollama_status
            },
            "phi3": {
                "model": config.PHI3_MODEL,
                **phi3_status
            }
        },
        app={
            "todos_count": len(todos),
            "port": os.getenv("PORT", "8000"),
            "log_level": config.LOG_LEVEL
        }
    )

@app.get("/metrics")
async def get_metrics():
    """Application metrics for monitoring"""
    return {
        "todos": {
            "total": len(todos),
            "completed": len([t for t in todos if t.completed]),
            "pending": len([t for t in todos if not t.completed]),
            "by_priority": {
                "high": len([t for t in todos if t.priority == "high"]),
                "medium": len([t for t in todos if t.priority == "medium"]),
                "low": len([t for t in todos if t.priority == "low"])
            }
        },
        "system": {
            "ollama_healthy": ollama_service.is_healthy,
            "model_loaded": ollama_service.model_loaded,
            "environment": config.ENVIRONMENT
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    try:
        return FileResponse('frontend.html')
    except FileNotFoundError:
        return {
            "message": "🤖 Professional AI Todo App with Phi-3",
            "version": "3.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "chat": "/chat",
            "metrics": "/metrics"
        }

# Main execution
if __name__ == "__main__":
    # Environment-based configuration
    is_production = config.ENVIRONMENT == "production"
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    if is_production:
        logger.info(f"🚀 Starting PRODUCTION server on {host}:{port}")
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
    else:
        logger.info(f"🔧 Starting DEVELOPMENT server on {host}:{port}")
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )