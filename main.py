import os
import requests
import time
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434"
PHI3_MODEL = "phi3:latest"

# Global storage (replace with DB in production)
todos = []
todo_counter = 1


# ---------------- Pydantic Models ----------------
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
    model_used: str
    timestamp: str
    action_performed: Optional[str] = None
    todos_affected: Optional[List[Todo]] = None

    model_config = ConfigDict(protected_namespaces=())  # Fix Pydantic warning


# ---------------- Ollama Utils ----------------
def wait_for_ollama(max_retries: int = 60) -> bool:
    """Wait for Ollama service to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama is ready at {OLLAMA_URL}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for Ollama... ({i+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"❓ Error: {e}")
            time.sleep(2)

    print("❌ Failed to connect to Ollama after retries")
    return False


def check_phi3_model() -> bool:
    """Check if Phi-3 model is available"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any("phi3" in name.lower() for name in model_names)
    except Exception as e:
        print(f"❌ Error checking Phi-3 model: {e}")
    return False


def call_phi3(prompt: str, system_prompt: str = None) -> str:
    """Call the Phi-3 model via Ollama"""
    try:
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:" if system_prompt else prompt

        payload = {
            "model": PHI3_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 4096, "num_predict": 256},
        }

        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip() or "No response from Phi-3."
        return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Phi-3 request timed out."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. The service might be starting."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ---------------- Lifespan (Startup + Shutdown) ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AI Todo App with Phi-3...")
    if wait_for_ollama() and check_phi3_model():
        test_response = call_phi3("Hello Phi-3, are you working?")
        print(f"✅ Phi-3 test response: {test_response[:60]}...")
    else:
        print("⚠️ Ollama or Phi-3 not ready.")
    yield
    print("🛑 Shutting down app...")


# ---------------- FastAPI App ----------------
app = FastAPI(
    title="Real AI Todo App with Phi-3",
    description="A todo app powered by actual Phi-3 model via Ollama",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Todo CRUD ----------------
@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
    global todo_counter
    new_todo = Todo(
        id=todo_counter,
        title=todo.title,
        description=todo.description or "",
        completed=False,
        created_at=datetime.now().isoformat(),
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
            todos[i].description = updated_todo.description or ""
            todos[i].completed = updated_todo.completed
            return todos[i]
    raise HTTPException(status_code=404, detail="Todo not found")


# ---------------- AI Chat ----------------
@app.post("/chat", response_model=ChatResponse)
def chat_with_phi3(request: ChatRequest):
    system_prompt = f"""You are an AI assistant for a todo app.
Current todos ({len(todos)} total): {[t.title for t in todos[-5:]] or "None"}"""
    ai_response = call_phi3(request.message, system_prompt)

    action_performed, todos_affected = None, []
    msg = request.message.lower()
    if "add" in msg and "todo" in msg:
        action_performed = "create_intent_detected"
    elif "list" in msg:
        action_performed = "list_intent_detected"
        todos_affected = todos.copy()
    elif "complete" in msg:
        action_performed = "complete_intent_detected"

    return ChatResponse(
        response=ai_response,
        model_used="phi3",
        timestamp=datetime.now().isoformat(),
        action_performed=action_performed,
        todos_affected=todos_affected,
    )


@app.post("/chat/direct")
def direct_phi3_chat(message: str):
    return {
        "user_message": message,
        "phi3_response": call_phi3(message),
        "model": "phi3",
        "timestamp": datetime.now().isoformat(),
    }


# ---------------- Health + Models ----------------
@app.get("/health")
def health_check():
    ollama_status = "connected" if wait_for_ollama(1) else "disconnected"
    phi3_status = "working" if check_phi3_model() else "not_found"
    return {
        "status": "healthy" if phi3_status == "working" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama_status": ollama_status,
        "phi3_status": phi3_status,
    }


@app.get("/models")
def list_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Root ----------------
@app.get("/")
def root():
    try:
        return FileResponse("frontend.html")
    except FileNotFoundError:
        return {"message": "✅ AI Todo App with Phi-3 is running!", "docs": "/docs"}
