import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import re
from datetime import datetime

# Create FastAPI app
app = FastAPI(title="AI-Powered Todo App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
todos = []
todo_counter = 1

# Models
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
    action_performed: Optional[str] = None
    todos_affected: Optional[List[Todo]] = None

# CRUD Operations
@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
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
    return new_todo

@app.get("/todos", response_model=List[Todo])
def get_todos():
    return todos

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    todo = next((t for t in todos if t.id == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated_todo: Todo):
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[i].title = updated_todo.title
            todos[i].description = updated_todo.description
            todos[i].completed = updated_todo.completed
            return todos[i]
    raise HTTPException(status_code=404, detail="Todo not found")

# AI Helper Functions
def call_phi3_local(prompt: str) -> str:
    """Try to call local Ollama first"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json()["response"]
    except:
        pass
    return None

def call_phi3_mock(prompt: str) -> str:
    """Mock AI response for cloud deployment"""
    if "add" in prompt.lower() or "create" in prompt.lower():
        return "Great! I've noted that down for you. That sounds like an important task!"
    elif "list" in prompt.lower() or "show" in prompt.lower():
        return "Here are your current tasks. You're doing great managing your todos!"
    elif "complete" in prompt.lower() or "done" in prompt.lower():
        return "Awesome! Well done on completing that task. Keep up the good work!"
    else:
        return "I'm here to help you stay organized and productive! What would you like to work on today?"

def call_phi3(prompt: str) -> str:
    """Smart AI caller - tries local Ollama first, falls back to mock"""
    # Try local Ollama first (for development)
    local_response = call_phi3_local(prompt)
    if local_response:
        return local_response
    
    # Fallback to mock (for production)
    return call_phi3_mock(prompt)

def extract_todo_info(text: str) -> dict:
    """Extract todo information from natural language"""
    patterns = [
        r"add.*?todo.*?[\"']([^\"']+)[\"']",
        r"create.*?task.*?[\"']([^\"']+)[\"']",
        r"add.*?[\"']([^\"']+)[\"']"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return {"title": match.group(1)}
    
    # Extract after keywords
    keywords = ["add", "create", "todo", "task"]
    words = text.lower().split()
    
    for i, word in enumerate(words):
        if word in keywords and i + 1 < len(words):
            title = " ".join(words[i+1:])
            title = re.sub(r'\b(a|an|the|to|for|me)\b', '', title).strip()
            if title:
                return {"title": title}
    
    return {}

def parse_user_intent(message: str) -> dict:
    """Parse user message to understand intent"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["add", "create", "new", "make"]):
        todo_info = extract_todo_info(message)
        return {"intent": "create_todo", "todo_info": todo_info}
    elif any(word in message_lower for word in ["show", "list", "get", "see", "display"]):
        return {"intent": "list_todos"}
    elif any(word in message_lower for word in ["complete", "done", "finish", "mark"]):
        return {"intent": "complete_todo"}
    else:
        return {"intent": "general_chat"}

# Main Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_with_ai(request: ChatRequest):
    """Main chat endpoint"""
    intent_data = parse_user_intent(request.message)
    intent = intent_data["intent"]
    
    if intent == "create_todo":
        todo_info = intent_data.get("todo_info", {})
        if "title" in todo_info:
            new_todo = create_todo(Todo(title=todo_info["title"]))
            return ChatResponse(
                response=f"Great! I've created a new todo: '{new_todo.title}' with ID {new_todo.id}",
                action_performed="create_todo",
                todos_affected=[new_todo]
            )
        else:
            return ChatResponse(
                response="I'd love to help you create a todo! Could you tell me what task you'd like to add?"
            )
    
    elif intent == "list_todos":
        all_todos = get_todos()
        if not all_todos:
            return ChatResponse(
                response="You don't have any todos yet. Would you like to create one?",
                action_performed="list_todos",
                todos_affected=[]
            )
        
        todo_list = "Here are your current todos:\n"
        for todo in all_todos:
            status = "✅" if todo.completed else "⏳"
            todo_list += f"{status} {todo.id}: {todo.title}\n"
        
        return ChatResponse(
            response=todo_list,
            action_performed="list_todos",
            todos_affected=all_todos
        )
    
    elif intent == "complete_todo":
        numbers = re.findall(r'\d+', request.message)
        if numbers:
            todo_id = int(numbers[0])
            try:
                todo = get_todo(todo_id)
                todo.completed = True
                updated_todo = update_todo(todo_id, todo)
                return ChatResponse(
                    response=f"Awesome! I've marked '{updated_todo.title}' as completed! 🎉",
                    action_performed="complete_todo",
                    todos_affected=[updated_todo]
                )
            except HTTPException:
                return ChatResponse(
                    response=f"I couldn't find a todo with ID {todo_id}. Use 'show todos' to see all your tasks."
                )
        else:
            return ChatResponse(
                response="Which todo would you like to mark as complete? Please include the todo ID number."
            )
    
    else:
        context = f"""
        You are a helpful assistant for a todo application. The user said: "{request.message}"
        Current todos in the system: {len(todos)} todos
        You can help users with managing their tasks effectively.
        """
        ai_response = call_phi3(context)
        return ChatResponse(response=ai_response)

# Serve static files
@app.get("/")
def serve_frontend():
    """Serve the frontend HTML"""
    try:
        return FileResponse('frontend.html')
    except:
        return {"message": "AI Todo App is running!", "environment": "production" if os.environ.get("PORT") else "development"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "todos_count": len(todos), 
        "environment": "production" if os.environ.get("PORT") else "development",
        "ollama_available": call_phi3_local("test") is not None
    }

# Smart host/port configuration for both environments
if __name__ == "__main__":
    import uvicorn
    
    # Check if running in production (Railway sets PORT env var)
    is_production = os.environ.get("PORT") is not None
    
    if is_production:
        # Production configuration (Railway, Render, etc.)
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"
        print(f"🚀 Starting PRODUCTION server on {host}:{port}")
    else:
        # Local development configuration
        port = 8000
        host = "localhost"  # or "127.0.0.1"
        print(f"💻 Starting LOCAL server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=not is_production  # Enable reload only in development
    )