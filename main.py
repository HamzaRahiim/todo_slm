import os
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Real AI Todo with Phi-3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def wait_for_ollama():
    """Wait for Ollama to be ready"""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama is ready at {OLLAMA_URL}")
                return True
        except:
            print(f"⏳ Waiting for Ollama... ({i+1}/{max_retries})")
            time.sleep(2)
    return False

def call_real_phi3(prompt: str) -> str:
    """Call actual Phi-3 model via Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 2048
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Ollama returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error: Phi-3 response timed out"
    except Exception as e:
        return f"Error connecting to Phi-3: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Wait for Ollama on startup"""
    print("🚀 Starting FastAPI with real Phi-3...")
    if not wait_for_ollama():
        print("❌ Failed to connect to Ollama!")
    else:
        # Test Phi-3
        test_response = call_real_phi3("Hello, are you working?")
        print(f"🧠 Phi-3 test response: {test_response[:50]}...")

@app.post("/chat")
def chat_with_real_phi3(message: str):
    """Direct chat with Phi-3 - no regex, pure AI"""
    
    # Create context for Phi-3
    context = f"""
You are an AI assistant for a todo application. The user said: "{message}"

You can help users by understanding their natural language and responding appropriately.
If they want to add a todo, extract the task and respond naturally.
If they want to list todos, ask what they want to see.
If they want to complete todos, ask which one.

Be conversational and helpful.

User: {message}
Assistant: """
    
    # Get pure AI response
    ai_response = call_real_phi3(context)
    
    return {
        "response": ai_response,
        "model": "phi3",
        "source": "real_ollama"
    }

@app.get("/health")
def health_check():
    # Test if Phi-3 is really working
    test_response = call_real_phi3("Say 'working' if you can respond.")
    
    return {
        "status": "healthy",
        "ollama_url": OLLAMA_URL,
        "phi3_test": test_response[:100] + "..." if len(test_response) > 100 else test_response,
        "model_available": "working" in test_response.lower()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
