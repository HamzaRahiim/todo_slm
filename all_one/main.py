# main.py - Multi-AI FastAPI Application
import os
import io
import base64
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-AI App",
    description="Text Generation + Image Generation + Voice Synthesis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100

class ImageRequest(BaseModel):
    prompt: str
    style: Optional[str] = "realistic"  # realistic, artistic, cartoon

class VoiceRequest(BaseModel):
    text: str
    language: Optional[str] = "en"  # en, es, fr, de, etc.

class MultiAIRequest(BaseModel):
    prompt: str
    generate_text: bool = True
    generate_image: bool = True
    generate_voice: bool = True

# AI Service Class
class MultiAIService:
    def __init__(self):
        # API Keys from environment variables
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN") 
        self.google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Initialize services
        self._init_gemini()
        self._init_huggingface()
        self._init_google_tts()
    
    def _init_gemini(self):
        """Initialize Google Gemini"""
        try:
            if self.gemini_key:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini initialized")
            else:
                logger.warning("⚠️ Gemini API key not found")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.gemini_model = None
    
    def _init_huggingface(self):
        """Initialize Hugging Face"""
        try:
            if self.hf_token:
                import requests
                self.hf_headers = {"Authorization": f"Bearer {self.hf_token}"}
                # Test API
                test_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
                response = requests.get(test_url, headers=self.hf_headers)
                if response.status_code == 200:
                    logger.info("✅ Hugging Face initialized")
                    self.hf_available = True
                else:
                    logger.warning(f"⚠️ Hugging Face API test failed: {response.status_code}")
                    self.hf_available = False
            else:
                logger.warning("⚠️ Hugging Face token not found")
                self.hf_available = False
        except Exception as e:
            logger.error(f"❌ Hugging Face initialization failed: {e}")
            self.hf_available = False
    
    def _init_google_tts(self):
        """Initialize Google Text-to-Speech"""
        try:
            if self.google_creds_path and os.path.exists(self.google_creds_path):
                from google.cloud import texttospeech
                self.tts_client = texttospeech.TextToSpeechClient()
                logger.info("✅ Google TTS initialized")
            else:
                # Fallback to gTTS (free alternative)
                logger.info("🔄 Using gTTS as fallback")
                self.tts_client = "gtts"
        except Exception as e:
            logger.error(f"❌ Google TTS initialization failed: {e}")
            self.tts_client = "gtts"
    
    async def generate_text(self, prompt: str, max_tokens: int = 100) -> dict:
        """Generate text using Gemini"""
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
                
                return {
                    "success": True,
                    "text": response.text,
                    "provider": "Google Gemini",
                    "tokens_used": len(response.text.split())
                }
            else:
                # Fallback response
                return {
                    "success": False,
                    "text": f"Mock response for: '{prompt}'. Gemini not available.",
                    "provider": "Fallback",
                    "error": "Gemini API key not configured"
                }
        
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return {
                "success": False,
                "text": f"Error generating text: {str(e)}",
                "provider": "Error",
                "error": str(e)
            }
    
    async def generate_image(self, prompt: str, style: str = "realistic") -> dict:
        """Generate image using Hugging Face Stable Diffusion"""
        try:
            if not self.hf_available:
                return {
                    "success": False,
                    "message": "Hugging Face not available",
                    "error": "HF token not configured"
                }
            
            import requests
            
            # Enhance prompt based on style
            style_prompts = {
                "realistic": f"{prompt}, photorealistic, high quality, detailed",
                "artistic": f"{prompt}, artistic, painting style, beautiful",
                "cartoon": f"{prompt}, cartoon style, colorful, animated"
            }
            
            enhanced_prompt = style_prompts.get(style, prompt)
            
            # Hugging Face API call
            api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            
            response = requests.post(
                api_url,
                headers=self.hf_headers,
                json={"inputs": enhanced_prompt},
                timeout=60
            )
            
            if response.status_code == 200:
                # Save image
                image_filename = f"generated_image_{hash(prompt) % 10000}.png"
                image_path = f"static/{image_filename}"
                
                # Create static directory if it doesn't exist
                os.makedirs("static", exist_ok=True)
                
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                return {
                    "success": True,
                    "image_path": image_path,
                    "image_url": f"/static/{image_filename}",
                    "provider": "Hugging Face Stable Diffusion",
                    "prompt_used": enhanced_prompt
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Image generation failed: {response.status_code}",
                    "error": response.text
                }
        
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return {
                "success": False,
                "message": f"Error generating image: {str(e)}",
                "error": str(e)
            }
    
    async def generate_voice(self, text: str, language: str = "en") -> dict:
        """Generate voice using Google TTS or gTTS fallback"""
        try:
            voice_filename = f"generated_voice_{hash(text) % 10000}.mp3"
            voice_path = f"static/{voice_filename}"
            
            # Create static directory if it doesn't exist
            os.makedirs("static", exist_ok=True)
            
            if self.tts_client != "gtts" and hasattr(self.tts_client, 'synthesize_speech'):
                # Use Google Cloud TTS
                from google.cloud import texttospeech
                
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language,
                    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                with open(voice_path, "wb") as f:
                    f.write(response.audio_content)
                
                provider = "Google Cloud TTS"
            
            else:
                # Use gTTS fallback
                from gtts import gTTS
                
                tts = gTTS(text=text, lang=language)
                tts.save(voice_path)
                
                provider = "gTTS (Free)"
            
            return {
                "success": True,
                "voice_path": voice_path,
                "voice_url": f"/static/{voice_filename}",
                "provider": provider,
                "text_length": len(text)
            }
        
        except Exception as e:
            logger.error(f"Voice generation error: {e}")
            return {
                "success": False,
                "message": f"Error generating voice: {str(e)}",
                "error": str(e)
            }

# Initialize AI service
ai_service = MultiAIService()

# Routes
@app.get("/")
async def root():
    return {
        "message": "🤖 Multi-AI FastAPI App",
        "services": ["Text Generation", "Image Generation", "Voice Synthesis"],
        "endpoints": {
            "text": "/generate-text",
            "image": "/generate-image", 
            "voice": "/generate-voice",
            "multi": "/generate-multi",
            "docs": "/docs"
        },
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "gemini": "available" if ai_service.gemini_model else "unavailable",
            "huggingface": "available" if ai_service.hf_available else "unavailable", 
            "google_tts": "available" if ai_service.tts_client != "gtts" else "fallback"
        }
    }

@app.post("/generate-text")
async def generate_text_endpoint(request: TextRequest):
    """Generate text using Gemini"""
    result = await ai_service.generate_text(request.prompt, request.max_tokens)
    return result

@app.post("/generate-image")
async def generate_image_endpoint(request: ImageRequest):
    """Generate image using Hugging Face"""
    result = await ai_service.generate_image(request.prompt, request.style)
    return result

@app.post("/generate-voice")
async def generate_voice_endpoint(request: VoiceRequest):
    """Generate voice using Google TTS"""
    result = await ai_service.generate_voice(request.text, request.language)
    return result

@app.post("/generate-multi")
async def generate_multi_endpoint(request: MultiAIRequest):
    """Generate text, image, and voice from single prompt"""
    results = {}
    
    if request.generate_text:
        results["text"] = await ai_service.generate_text(request.prompt)
    
    if request.generate_image:
        results["image"] = await ai_service.generate_image(request.prompt)
    
    if request.generate_voice and request.generate_text:
        # Use generated text for voice, or original prompt
        voice_text = results["text"]["text"] if "text" in results and results["text"]["success"] else request.prompt
        results["voice"] = await ai_service.generate_voice(voice_text)
    
    return {
        "prompt": request.prompt,
        "results": results,
        "timestamp": "2024-01-01T12:00:00"  # You can add real timestamp
    }

# Serve static files (images, audio)
@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve generated images and audio files"""
    file_path = f"static/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Main execution
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting Multi-AI FastAPI App...")
    print(f"📱 Local: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🎯 Test endpoints:")
    print(f"  - Text: POST /generate-text")
    print(f"  - Image: POST /generate-image")  
    print(f"  - Voice: POST /generate-voice")
    print(f"  - Multi: POST /generate-multi")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=True
    )