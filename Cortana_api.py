from cortana_agent import Cortana_agent
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
from dotenv import load_dotenv
from pydantic_ai import BinaryContent
import os
import requests
import hashlib
load_dotenv()

import uvicorn
import time
import json

import logfire

app = FastAPI(
    title="Cortana API", 
    description="""
    ## Cortana AI Assistant API
    
    A comprehensive API for interacting with Cortana AI Assistant with multi-modal capabilities.
    
    ### Available Endpoints:
    
    **GET Requests:**
    - `/health` - Check API health status and uptime
    - `/docs` - Get comprehensive API documentation
    
    **POST Requests:**
    - `/chat` - Main chat endpoint with multi-modal support (text, image, voice, document uploads)
    - `/text-to-speech` - Convert text to speech using HuggingFace TTS
    - `/reset` - Reset Cortana's memory and conversation history
    
    ### Features:
    - Multi-modal input support (text, images, voice, documents)
    - Text-to-speech conversion
    - Web search integration via Google and Tavily
    - OpenAI GPT integration
    - Composio tools integration
    - Memory management and conversation reset
    - Health monitoring and uptime tracking
    """,
    version="0.1.0"
)
logfire.configure(token=os.getenv('logfire_token'))
logfire.instrument_fastapi(app)
# Initialize Cortana agent instance

startup_time = time.time()

class TTSResponse(BaseModel):
    text: str
    audio_url: str

class EndpointInfo(BaseModel):
    path: str
    method: str
    description: str
    parameters: List[Dict[str, str]]
    example_request: Dict[str, str]
    example_response: Dict[str, str]

class APIDocumentation(BaseModel):
    name: str
    version: str
    description: str
    endpoints: List[EndpointInfo]

class KeyCache:
    def __init__(self):
        self._last_keys_hash = None
        self._cortana = None
    
    def _compute_keys_hash(self, api_keys: Dict[str, str]) -> str:
        # Sort keys to ensure consistent hashing regardless of order
        sorted_keys = dict(sorted(api_keys.items()))
        # Create a string representation of the keys
        keys_str = "|".join(f"{k}:{v}" for k, v in sorted_keys.items() if v is not None)
        # Compute hash
        return hashlib.sha256(keys_str.encode()).hexdigest()
    
    def get_cortana(self, api_keys: Dict[str, str]) -> Cortana_agent:
        current_hash = self._compute_keys_hash(api_keys)
        
        # Initialize or reinitialize if keys have changed
        if self._last_keys_hash != current_hash:
            # Filter out None values for initialization
            init_keys = {k: v for k, v in api_keys.items() if v is not None}
            # Pass the entire dictionary as a single parameter
            self._cortana = Cortana_agent(api_keys=init_keys)
            self._last_keys_hash = current_hash
        
        return self._cortana
    def reset(self):
        self._cortana.reset()
        self._last_keys_hash = None
# Initialize key cache
key_cache = KeyCache()

def get_tts_audio(text: str, hf_token: str) -> str:
    API_URL = "https://router.huggingface.co/fal-ai/fal-ai/kokoro/american-english"
    headers = {
        "Authorization": f"Bearer {hf_token}",
    }
    response = requests.post(API_URL, headers=headers, json={"text": text})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate speech")
    return response.json().get("audio", {}).get("url")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "uptime": time.time() - startup_time,
        "version": "0.1.0",
        "service": "Cortana API"
    }

@app.post("/chat")
async def chat(
    query: str = Form(...),
    google_api_key: str = Form(...),
    tavily_key: str = Form(...),
    pse: str = Form(None),
    openai_api_key: str = Form(None),
    composio_key: str = Form(None),
    hf_token: str = Form(None),
    include_audio: bool = Form(False),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None),
):
    try:
        # Create a list of inputs starting with the query
        inputs = [query]
        
        # Add any uploaded files as BinaryContent objects
        for file_type, file_obj in [("image", image), ("voice", voice), ("document", document)]:
            if file_obj is not None:
                contents = await file_obj.read()
                binary_content = BinaryContent(data=contents, media_type=file_obj.content_type)
                
                inputs = [query, binary_content]
                await file_obj.close()

        api_keys = {
            "google_api_key": google_api_key,
            "tavily_key": tavily_key,
            "openai_api_key": openai_api_key,
            "pse": pse,
            "composio_key": composio_key
        }
        
        # Get or initialize Cortana instance based on keys
        try:
            cortana = key_cache.get_cortana(api_keys)
            
        except Exception as e:
            
            raise HTTPException(status_code=500, detail=f"Error getting Cortana instance: {str(e)}")
        response = cortana.chat(inputs)
        # Generate audio if requested and token provided
        audio_url = None
        if include_audio and hf_token:
            try:
                audio_url = get_tts_audio(response.ui_version, hf_token)
            except Exception as e:
                print(f"TTS generation failed: {str(e)}")
        
        return {
            "response": response.ui_version,
            "audio_url": audio_url
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/reset")
async def reset_cortana():
    try:
        key_cache.reset()
        return {"status": "success", "message": "Cortana memory reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    hf_token: str = Form(...),
):
    """
    Convert text to speech using HuggingFace's TTS API.
    Returns a URL to the generated audio file.
    """
    try:
        audio_url = get_tts_audio(text, hf_token)
        return {"audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", response_model=APIDocumentation)
async def get_documentation():
    """
    Returns comprehensive documentation for all API endpoints
    """
    return APIDocumentation(
        name="Cortana API",
        version="0.1.0",
        description="API for interacting with Cortana AI Assistant, including chat, text-to-speech, and file processing capabilities",
        endpoints=[
            EndpointInfo(
                path="/chat",
                method="POST",
                description="Main chat endpoint that processes text queries and optional file uploads. Can include text-to-speech conversion.",
                parameters=[
                    {"name": "query", "type": "string", "required": "Yes", "description": "The text query to process"},
                    {"name": "google_api_key", "type": "string", "required": "Yes", "description": "Google API key for search functionality"},
                    {"name": "tavily_key", "type": "string", "required": "Yes", "description": "Tavily API key for search functionality"},
                    {"name": "pse", "type": "string", "required": "No", "description": "Personal search engine identifier"},
                    {"name": "openai_api_key", "type": "string", "required": "No", "description": "OpenAI API key for language model"},
                    {"name": "composio_key", "type": "string", "required": "No", "description": "Composio API key"},
                    {"name": "hf_token", "type": "string", "required": "No", "description": "HuggingFace token for text-to-speech"},
                    {"name": "include_audio", "type": "boolean", "required": "No", "description": "Whether to include audio in response"},
                    {"name": "image", "type": "file", "required": "No", "description": "Optional image file upload"},
                    {"name": "voice", "type": "file", "required": "No", "description": "Optional voice file upload"},
                    {"name": "document", "type": "file", "required": "No", "description": "Optional document file upload"}
                ],
                example_request={
                    "query": "What's the weather like?",
                    "google_api_key": "your_google_api_key",
                    "tavily_key": "your_tavily_key",
                    "include_audio": "true",
                    "hf_token": "your_hf_token"
                },
                example_response={
                    "response": "The weather is currently sunny with a temperature of 72°F.",
                    "audio_url": "https://huggingface.co/audio/..."
                }
            ),
            EndpointInfo(
                path="/text-to-speech",
                method="POST",
                description="Convert text to speech using HuggingFace's TTS API",
                parameters=[
                    {"name": "text", "type": "string", "required": "Yes", "description": "Text to convert to speech"},
                    {"name": "hf_token", "type": "string", "required": "Yes", "description": "HuggingFace API token"}
                ],
                example_request={
                    "text": "Hello, this is a test",
                    "hf_token": "your_hf_token"
                },
                example_response={
                    "audio_url": "https://huggingface.co/audio/..."
                }
            ),
            EndpointInfo(
                path="/reset",
                method="POST",
                description="Reset Cortana's memory and conversation history",
                parameters=[],
                example_request={},
                example_response={
                    "status": "success",
                    "message": "Cortana memory reset successfully"
                }
            ),
            EndpointInfo(
                path="/health",
                method="GET",
                description="Check API health status and uptime",
                parameters=[],
                example_request={},
                example_response={
                    "status": "healthy",
                    "uptime": 3600.5,
                    "version": "0.1.0",
                    "service": "Cortana API"
                }
            )
        ]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

