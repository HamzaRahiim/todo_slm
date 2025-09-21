# simple_main.py - Simplified Multi-AI App (No complex dependencies)
import os
import io
import requests
import hashlib
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Simple data models (without pydantic validation for now)
app = FastAPI(title="Simple Multi-AI App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "your-hf-token")

# Create static directory
os.makedirs("static", exist_ok=True)

class SimpleAIService:
    def __init__(self):
        self.gemini_available = bool(GEMINI_API_KEY != "your-gemini-key")
        self.hf_available = bool(HF_TOKEN != "your-hf-token")
    
    def generate_text_gemini(self, prompt: str):
        """Generate text using Gemini"""
        try:
            if not self.gemini_available:
                return {
                    "success": False,
                    "text": f"Mock response: '{prompt}' - Gemini API key needed",
                    "provider": "Mock"
                }
            
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return {
                "success": True,
                "text": response.text,
                "provider": "Google Gemini"
            }
        
        except Exception as e:
            return {
                "success": False,
                "text": f"Error: {str(e)}",
                "provider": "Error"
            }
    
    def generate_image_hf(self, prompt: str):
        """Generate image using Hugging Face"""
        try:
            if not self.hf_available:
                return {
                    "success": False,
                    "message": "Hugging Face token needed",
                    "image_url": None
                }
            
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": f"{prompt}, high quality, detailed"},
                timeout=60
            )
            
            if response.status_code == 200:
                # Generate filename
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                filename = f"image_{prompt_hash}.png"
                filepath = f"static/{filename}"
                
                # Save image
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                return {
                    "success": True,
                    "message": "Image generated successfully",
                    "image_url": f"/static/{filename}",
                    "filename": filename
                }
            else:
                return {
                    "success": False,
                    "message": f"API error: {response.status_code}",
                    "error": response.text[:200]
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def generate_voice_gtts(self, text: str):
        """Generate voice using gTTS"""
        try:
            from gtts import gTTS
            
            # Generate filename
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"voice_{text_hash}.mp3"
            filepath = f"static/{filename}"
            
            # Generate voice
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)
            
            return {
                "success": True,
                "message": "Voice generated successfully",
                "voice_url": f"/static/{filename}",
                "filename": filename,
                "text_length": len(text)
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }

# Initialize service
ai_service = SimpleAIService()

# Routes
@app.get("/")
def root():
    return {
        "message": "🤖 Simple Multi-AI App",
        "services": {
            "gemini": "available" if ai_service.gemini_available else "needs_api_key",
            "huggingface": "available" if ai_service.hf_available else "needs_token",
            "gtts": "available"
        },
        "endpoints": {
            "text": "/text",
            "image": "/image",
            "voice": "/voice",
            "all": "/generate-all",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_ready": ai_service.gemini_available,
        "hf_ready": ai_service.hf_available,
        "gtts_ready": True
    }

@app.post("/text")
def generate_text(prompt: str):
    """Generate text using Gemini"""
    result = ai_service.generate_text_gemini(prompt)
    return result

@app.post("/image")
def generate_image(prompt: str):
    """Generate image using Hugging Face"""
    result = ai_service.generate_image_hf(prompt)
    return result

@app.post("/voice")
def generate_voice(text: str):
    """Generate voice using gTTS"""
    result = ai_service.generate_voice_gtts(text)
    return result

@app.post("/generate-all")
def generate_all(prompt: str):
    """Generate text, image, and voice from one prompt"""
    
    # Generate text first
    text_result = ai_service.generate_text_gemini(prompt)
    
    # Generate image
    image_result = ai_service.generate_image_hf(prompt)
    
    # Generate voice from the generated text or original prompt
    voice_text = text_result.get("text", prompt) if text_result.get("success") else prompt
    voice_result = ai_service.generate_voice_gtts(voice_text[:500])  # Limit text length
    
    return {
        "prompt": prompt,
        "results": {
            "text": text_result,
            "image": image_result,
            "voice": voice_result
        },
        "summary": {
            "text_success": text_result.get("success", False),
            "image_success": image_result.get("success", False),
            "voice_success": voice_result.get("success", False)
        }
    }

# Serve static files
@app.get("/static/{filename}")
def serve_static(filename: str):
    filepath = f"static/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Test endpoints (for easy browser testing)
@app.get("/test-text")
def test_text():
    return ai_service.generate_text_gemini("Write a short poem about coding")

@app.get("/test-image")  
def test_image():
    return ai_service.generate_image_hf("A beautiful sunset over mountains")

@app.get("/test-voice")
def test_voice():
    return ai_service.generate_voice_gtts("Hello! This is a test of voice synthesis.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    
    print("🚀 Starting Simple Multi-AI App...")
    print(f"📱 Local: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🧪 Quick tests:")
    print(f"  - Text: http://localhost:{port}/test-text")
    print(f"  - Image: http://localhost:{port}/test-image")
    print(f"  - Voice: http://localhost:{port}/test-voice")
    
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )