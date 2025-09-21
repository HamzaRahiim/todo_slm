import os
import requests
import time
import json
import re
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(
    title="Real AI Todo App with Phi-3",
    description="A todo app powered by actual Phi-3 model via Ollama",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434"
PHI3_MODEL = "phi3:latest"

# Global storage (in production, you'd use a database)
todos = []
todo_counter = 1

# Pydantic Models
class Todo(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    completed: bool = False
    created_at: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    model_used: str = "phi3"
    timestamp: str = datetime.now().isoformat()
    action_performed: Optional[str] = None
    todos_affected: Optional[List[Todo]] = None

# Ollama Connection Functions
def wait_for_ollama(max_retries: int = 60) -> bool:
    """Wait for Ollama service to be ready"""
    print("???? Checking Ollama connection...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama is ready at {OLLAMA_URL}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for Ollama to start... ({i+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"❓ Unexpected error: {e}")
            time.sleep(2)
    
    print("❌ Failed to connect to Ollama after maximum retries")
    return False

def check_phi3_model() -> bool:
    """Check if Phi-3 model is available"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            # Check for phi3 model (could be phi3:latest or just phi3)
            phi3_available = any("phi3" in name.lower() for name in model_names)
            
            if phi3_available:
                print("✅ Phi-3 model is available")
                return True
            else:
                print(f"❌ Phi-3 model not found. Available models: {model_names}")
                return False
    except Exception as e:
        print(f"❌ Error checking Phi-3 model: {e}")
        return False

def call_phi3(prompt: str, system_prompt: str = None) -> str:
    """Call the actual Phi-3 model via Ollama"""
    try:
        # Construct the full prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt

        # Make request to Ollama
        payload = {
            "model": PHI3_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 4096,  # Context length
                "num_predict": 256,  # Max tokens to generate
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        print(f"???? Sending request to Phi-3...")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60  # 60 seconds timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            if ai_response:
                print(f"✅ Phi-3 responded: {ai_response[:50]}...")
                return ai_response
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
        
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        error_msg = "Request to Phi-3 timed out. The model might be processing a complex request."
        print(f"⏰ {error_msg}")
        return error_msg
        
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Ollama. The service might be starting up."
        print(f"???? {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error calling Phi-3: {str(e)}"
        print(f"???? {error_msg}")
        return error_msg

# Todo CRUD Operations
@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
    """Create a new todo"""
    global todo_counter
    
    new_todo = Todo(
        id=todo_counter,
        title=todo.title,
        description=todo.description or "",
        completed=False,
        created_at=datetime.now().isoformat()
    )
    
    todos.append(new_todo)
    todo_counter += 1
    
    print(f"???? Created todo: {new_todo.title}")
    return new_todo

@app.get("/todos", response_model=List[Todo])
def get_todos():
    """Get all todos"""
    return todos

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    """Get a specific todo by ID"""
    todo = next((t for t in todos if t.id == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail=f"Todo with ID {todo_id} not found")
    return todo

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated_todo: Todo):
    """Update a todo"""
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[i].title = updated_todo.title
            todos[i].description = updated_todo.description or ""
            todos[i].completed = updated_todo.completed
            print(f"✏️ Updated todo {todo_id}: {updated_todo.title}")
            return todos[i]
    
    raise HTTPException(status_code=404, detail=f"Todo with ID {todo_id} not found")

# AI Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_with_phi3(request: ChatRequest):
    """Chat with Phi-3 for todo management"""
    
    # System prompt to give Phi-3 context about the todo app
    system_prompt = f"""You are an AI assistant for a todo application. You help users manage their tasks naturally.

Current todos in the system ({len(todos)} total):
{chr(10).join([f"- ID {todo.id}: {todo.title} ({'✅ Completed' if todo.completed else '⏳ Pending'})" for todo in todos[-5:]]) if todos else "No todos yet"}

You can help users with:
1. Understanding what they want to do with their todos
2. Providing helpful responses about task management
3. Giving productivity advice
4. Being encouraging and supportive

Please respond naturally and be helpful. If they mention adding, completing, or listing todos, acknowledge their request and be supportive."""

    # Get AI response
    ai_response = call_phi3(request.message, system_prompt)
    
    # Try to detect if user wants to perform actions (basic intent detection)
    message_lower = request.message.lower()
    action_performed = None
    todos_affected = []
    
    # Simple action detection (you can make this more sophisticated)
    if any(word in message_lower for word in ["add", "create", "new"]) and any(word in message_lower for word in ["todo", "task"]):
        action_performed = "create_intent_detected"
    elif any(word in message_lower for word in ["show", "list", "display"]):
        action_performed = "list_intent_detected"
        todos_affected = todos.copy()
    elif any(word in message_lower for word in ["complete", "done", "finish"]):
        action_performed = "complete_intent_detected"
    
    return ChatResponse(
        response=ai_response,
        model_used="phi3",
        action_performed=action_performed,
        todos_affected=todos_affected
    )

# Direct Phi-3 chat endpoint (for testing)
@app.post("/chat/direct")
def direct_phi3_chat(message: str):
    """Direct chat with Phi-3 without todo context"""
    response = call_phi3(message)
    
    return {
        "user_message": message,
        "phi3_response": response,
        "model": "phi3",
        "timestamp": datetime.now().isoformat()
    }

# Health and Status Endpoints
@app.get("/")
def root():
    """Root endpoint"""
    try:
        return FileResponse('frontend.html')
    except FileNotFoundError:
        return {
            "message": "???? AI Todo App with Phi-3 is running!",
            "docs": "/docs",
            "health": "/health",
            "chat": "/chat"
        }

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    
    # Check Ollama connection
    ollama_status = "unknown"
    ollama_error = None
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "connected"
        else:
            ollama_status = f"error_{response.status_code}"
    except Exception as e:
        ollama_status = "disconnected"
        ollama_error = str(e)
    
    # Check Phi-3 model
    phi3_status = "unknown"
    phi3_test = None
    
    if ollama_status == "connected":
        phi3_available = check_phi3_model()
        if phi3_available:
            # Test Phi-3 with a simple prompt
            test_response = call_phi3("Say 'Hello from Phi-3!' if you are working correctly.")
            phi3_test = test_response[:100] + "..." if len(test_response) > 100 else test_response
            
            if "hello from phi-3" in test_response.lower() or len(test_response) > 10:
                phi3_status = "working"
            else:
                phi3_status = "responding_but_unclear"
        else:
            phi3_status = "model_not_found"
    else:
        phi3_status = "ollama_not_connected"
    
    return {
        "status": "healthy" if phi3_status == "working" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "environment": "production" if os.environ.get("PORT") else "development",
        "ollama": {
            "url": OLLAMA_URL,
            "status": ollama_status,
            "error": ollama_error
        },
        "phi3": {
            "model": PHI3_MODEL,
            "status": phi3_status,
            "test_response": phi3_test
        },
        "app": {
            "todos_count": len(todos),
            "port": os.environ.get("PORT", "8000")
        }
    }

@app.get("/models")
def list_models():
    """List available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get models: {response.status_code}"}
    except Exception as e:
        return {"error": f"Cannot connect to Ollama: {str(e)}"}

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("???? Starting AI Todo App with real Phi-3...")
    print(f"???? Ollama URL: {OLLAMA_URL}")
    print(f"???? Model: {PHI3_MODEL}")
    
    # Wait for Ollama to be ready
    if wait_for_ollama():
        # Check if Phi-3 model is available
        if check_phi3_model():
            # Test Phi-3
            print("???? Testing Phi-3 model...")
            test_response = call_phi3("Hello! Please confirm you are Phi-3 and you are working correctly.")
            print(f"???? Phi-3 response: {test_response}")
            print("✅ Phi-3 is ready for production!")
        else:
            print("❌ Phi-3 model is not available!")
            print("???? Make sure 'ollama pull phi3' was successful during container build")
    else:
        print("❌ Ollama is not responding!")
        print("???? Check if Ollama service started correctly in the container")

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Determine environment
    is_production = os.environ.get("PORT") is not None
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    if is_production:
        print(f"???? Starting PRODUCTION server on {host}:{port}")
    else:
        print(f"???? Starting DEVELOPMENT server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Always disable reload in container
        log_level="info"
    )