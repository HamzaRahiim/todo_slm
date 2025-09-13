from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import re
import os
from datetime import datetime

app = FastAPI(title="AI-Powered Todo App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (you can replace with PostgreSQL later)
todos = []
todo_counter = 1

# Pydantic models
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

# Basic CRUD endpoints
@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
    global todo_counter
    new_todo = Todo(
        id=todo_counter,
        title=todo.title,
        description=todo.description,
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
    todo = next((t for t in todos if t.id == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    todo.title = updated_todo.title
    todo.description = updated_todo.description
    todo.completed = updated_todo.completed
    return todo

# Helper functions for AI integration
def call_phi3_mock(prompt: str) -> str:
    """Mock AI response for demo purposes when Ollama isn't available"""
    responses = {
        "create": "Great! I've noted that down for you. That sounds like an important task!",
        "list": "Here are your current tasks. You're doing great managing your todos!",
        "complete": "Awesome! Well done on completing that task. Keep up the good work!",
        "general": "I'm here to help you stay organized and productive! What would you like to work on today?"
    }
    
    if "add" in prompt.lower() or "create" in prompt.lower():
        return responses["create"]
    elif "list" in prompt.lower() or "show" in prompt.lower():
        return responses["list"]
    elif "complete" in prompt.lower() or "done" in prompt.lower():
        return responses["complete"]
    else:
        return responses["general"]

def call_phi3(prompt: str) -> str:
    """Call Phi-3 through Ollama API with fallback"""
    try:
        # Try to connect to local Ollama first
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["response"]
    except:
        pass
    
    # Fallback to mock responses for demo
    return call_phi3_mock(prompt)

def extract_todo_info(text: str) -> dict:
    """Extract todo information from natural language"""
    # Simple patterns to extract todo details
    title_patterns = [
        r"add.*?todo.*?[\"']([^\"']+)[\"']",
        r"create.*?task.*?[\"']([^\"']+)[\"']",
        r"add.*?task.*?[\"']([^\"']+)[\"']",
        r"create.*?[\"']([^\"']+)[\"']",
        r"add.*?[\"']([^\"']+)[\"']"
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text.lower())
        if match:
            return {"title": match.group(1)}
    
    # If no quotes, try to extract after keywords
    keywords = ["add", "create", "todo", "task"]
    words = text.lower().split()
    
    for i, word in enumerate(words):
        if word in keywords and i + 1 < len(words):
            # Take the rest as title
            title = " ".join(words[i+1:])
            # Clean common words
            title = re.sub(r'\b(a|an|the|to|for|me)\b', '', title).strip()
            if title:
                return {"title": title}
    
    return {}

def parse_user_intent(message: str) -> dict:
    """Parse user message to understand intent"""
    message_lower = message.lower()
    
    # Intent classification
    if any(word in message_lower for word in ["add", "create", "new", "make"]):
        if "todo" in message_lower or "task" in message_lower:
            todo_info = extract_todo_info(message)
            return {
                "intent": "create_todo",
                "todo_info": todo_info
            }
    
    elif any(word in message_lower for word in ["show", "list", "get", "see", "display"]):
        return {"intent": "list_todos"}
    
    elif any(word in message_lower for word in ["complete", "done", "finish", "mark"]):
        return {"intent": "complete_todo"}
    
    elif any(word in message_lower for word in ["update", "edit", "change", "modify"]):
        return {"intent": "update_todo"}
    
    else:
        return {"intent": "general_chat"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_ai(request: ChatRequest):
    """Main chat endpoint that processes user requests"""
    
    # Parse user intent
    intent_data = parse_user_intent(request.message)
    intent = intent_data["intent"]
    
    if intent == "create_todo":
        # Handle todo creation
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
                response="I'd love to help you create a todo! Could you tell me what task you'd like to add? For example: 'Add todo: Buy groceries'"
            )
    
    elif intent == "list_todos":
        # Handle listing todos
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
        # Handle marking todos as complete
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
        # General chat - pass to AI
        context = f"""
        You are a helpful assistant for a todo application. The user said: "{request.message}"
        
        Current todos in the system: {len(todos)} todos
        
        You can help users with:
        - Creating todos (say "add todo: task name")
        - Listing todos (say "show todos")
        - Marking todos complete (say "complete todo 1")
        - General questions about their tasks
        
        Respond helpfully and encourage them to manage their tasks.
        """
        
        ai_response = call_phi3(context)
        return ChatResponse(response=ai_response)

# Serve the frontend
@app.get("/")
def serve_frontend():
    return FileResponse('frontend.html')

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "todos_count": len(todos)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)