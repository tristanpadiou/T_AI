from cortana_agent import Cortana_agent
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse
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
    - `/api-docs` - Get comprehensive API documentation
    
    **POST Requests:**
    - `/chat` - Main chat endpoint with multi-modal support (text, multiple images, voice, document uploads)
    - `/text-to-speech` - Convert text to speech using HuggingFace TTS
    - `/reset` - Reset Cortana's memory and conversation history
    
    ### Features:
    - Multi-modal input support (text, multiple images, voice, documents)
    - Text-to-speech conversion
    - Web search integration via Google and Tavily
    - OpenAI GPT integration
    - Composio tools integration
    - Code execution integration
    - Image search integration
    - Memory management and conversation reset
    - Health monitoring and uptime tracking
    """,
    version="0.1.0",
    docs_url=None,  # Disable built-in docs
    redoc_url=None  # Disable redoc as well
)
logfire.configure(token=os.getenv('cortana_api_logfire_token'))
logfire.instrument_pydantic_ai()
# Initialize Cortana agent instance
logfire.instrument_fastapi(app)
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
    images: Optional[List[UploadFile]] = File(None),
    voice: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None),
):
    try:
        # Create a list of inputs starting with the query
        inputs = [query]
        
        # Handle multiple images
        if images:
            for image in images:
                if image is not None:
                    contents = await image.read()
                    binary_content = BinaryContent(data=contents, media_type=image.content_type)
                    inputs.append(binary_content)
                    await image.close()
        
        # Handle voice and document uploads
        for file_type, file_obj in [("voice", voice), ("document", document)]:
            if file_obj is not None:
                contents = await file_obj.read()
                binary_content = BinaryContent(data=contents, media_type=file_obj.content_type)
                inputs.append(binary_content)
                await file_obj.close()

        api_keys = {
            "google_api_key": google_api_key,
            "tavily_key": tavily_key,
            "openai_api_key": openai_api_key,
            "pse": pse,
            "composio_key": composio_key,
            "hf_token": hf_token
        }
        
        # Get or initialize Cortana instance based on keys
        try:
            cortana = key_cache.get_cortana(api_keys)
            
        except Exception as e:
            
            raise HTTPException(status_code=500, detail=f"Error getting Cortana instance: {str(e)}")
        
        # Handle the pydantic_ai usage tracking bug temporarily
        try:
            response = cortana.chat(inputs)
        except TypeError as e:
            if "unsupported operand type(s) for +: 'int' and 'list'" in str(e):
                # This is a known pydantic_ai bug with usage tracking
                # For now, we'll retry once which often works
                try:
                    response = cortana.chat(inputs)
                except Exception as retry_e:
                    raise HTTPException(status_code=500, detail=f"Error in chat after retry (known pydantic_ai bug): {str(retry_e)}")
            else:
                raise HTTPException(status_code=500, detail=f"Type error in chat: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")
        # Generate audio if requested and token provided
        audio_url = None
        if include_audio and hf_token:
            try:
                audio_url = get_tts_audio(response.voice_version, hf_token)
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

@app.get("/api-docs")
async def get_markdown_documentation():
    """
    Returns comprehensive documentation for all API endpoints in markdown format
    """
  
    return """# Cortana API Documentation

**Version:** 0.1.0

## Description
API for interacting with Cortana AI Assistant, including chat, text-to-speech, and file processing capabilities.

---

## Endpoints

### POST `/chat`
**Description:** Main chat endpoint that processes text queries and optional file uploads. Can include text-to-speech conversion.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | Yes | The text query to process |
| google_api_key | string | Yes | Google API key for search functionality |
| tavily_key | string | Yes | Tavily API key for search functionality |
| pse | string | No | Personal search engine identifier |
| openai_api_key | string | No | OpenAI API key for language model |
| composio_key | string | No | Composio API key |
| hf_token | string | No | HuggingFace token for text-to-speech |
| include_audio | boolean | No | Whether to include audio in response |
| images | file[] | No | Optional multiple image files upload |
| voice | file | No | Optional voice file upload |
| document | file | No | Optional document file upload |

**Example Request:**
```json
{
    "query": "What's the weather like?",
    "google_api_key": "your_google_api_key",
    "tavily_key": "your_tavily_key",
    "include_audio": "true",
    "hf_token": "your_hf_token"
}
```

**Example Response:**
```json
{
    "response": "The weather is currently sunny with a temperature of 72°F.",
    "audio_url": "https://huggingface.co/audio/..."
}
```

---

### POST `/text-to-speech`
**Description:** Convert text to speech using HuggingFace's TTS API

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| text | string | Yes | Text to convert to speech |
| hf_token | string | Yes | HuggingFace API token |

**Example Request:**
```json
{
    "text": "Hello, this is a test",
    "hf_token": "your_hf_token"
}
```

**Example Response:**
```json
{
    "audio_url": "https://huggingface.co/audio/..."
}
```

---

### POST `/reset`
**Description:** Reset Cortana's memory and conversation history

**Parameters:** None

**Example Request:**
```json

```

**Example Response:**
```json
{
    "status": "success",
    "message": "Cortana memory reset successfully"
}
```

---

### GET `/health`
**Description:** Check API health status and uptime

**Parameters:** None

**Example Request:** No body required

**Example Response:**
```json
{
    "status": "healthy",
    "uptime": 3600.5,
    "version": "0.1.0",
    "service": "Cortana API"
}
```

---

### GET `/docs`
**Description:** Get comprehensive API documentation in JSON format

**Parameters:** None

**Example Request:** No body required

**Example Response:** Returns structured JSON documentation

---

### GET `/docs/markdown`
**Description:** Get comprehensive API documentation in markdown format

**Parameters:** None

**Example Request:** No body required

**Example Response:** Returns this markdown documentation

---

## Features
- Multi-modal input support (text, images, voice, documents)
- Text-to-speech conversion
- Web search integration via Google and Tavily
- OpenAI GPT integration
- Composio tools integration
- Code execution integration
- Image search integration
- Memory management and conversation reset
- Health monitoring and uptime tracking

## Usage Notes
- All file uploads should use multipart/form-data encoding
- API keys are required for most functionality
- The chat endpoint supports multiple input types simultaneously
- Audio responses require a valid HuggingFace token
"""

@app.get("/docs")
async def get_docs():
    """
    Returns comprehensive documentation for all API endpoints in JSON format
    """
    return {
        "name": "Cortana API",
        "version": "0.1.0", 
        "description": "API for interacting with Cortana AI Assistant with multi-modal capabilities",
        "endpoints": [
            {
                "path": "/chat",
                "method": "POST",
                "description": "Main chat endpoint with multi-modal support for text, multiple images, voice, and documents",
                "content_type": "multipart/form-data",
                "parameters": [
                    {
                        "name": "query",
                        "type": "string",
                        "required": True,
                        "description": "The text query to process"
                    },
                    {
                        "name": "google_api_key", 
                        "type": "string",
                        "required": True,
                        "description": "Google API key for search functionality"
                    },
                    {
                        "name": "tavily_key",
                        "type": "string", 
                        "required": True,
                        "description": "Tavily API key for search functionality"
                    },
                    {
                        "name": "pse",
                        "type": "string",
                        "required": False,
                        "description": "Personal search engine identifier (optional)"
                    },
                    {
                        "name": "openai_api_key",
                        "type": "string",
                        "required": False,
                        "description": "OpenAI API key for language model (optional)"
                    },
                    {
                        "name": "composio_key", 
                        "type": "string",
                        "required": False,
                        "description": "Composio API key for tools integration (optional)"
                    },
                    {
                        "name": "hf_token",
                        "type": "string",
                        "required": False,
                        "description": "HuggingFace token for text-to-speech functionality (optional)"
                    },
                    {
                        "name": "include_audio",
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Whether to include audio response (requires hf_token)"
                    },
                    {
                        "name": "images",
                        "type": "file[]",
                        "required": False,
                        "description": "Optional multiple image files upload for visual analysis"
                    },
                    {
                        "name": "voice", 
                        "type": "file",
                        "required": False,
                        "description": "Optional voice/audio file upload for audio processing"
                    },
                    {
                        "name": "document",
                        "type": "file", 
                        "required": False,
                        "description": "Optional document file upload for document analysis"
                    }
                ],
                "response": {
                    "response": "string - The AI assistant's response",
                    "audio_url": "string - URL to generated audio (if include_audio=true and hf_token provided)"
                }
            },
            {
                "path": "/text-to-speech", 
                "method": "POST",
                "description": "Convert text to speech using HuggingFace TTS API",
                "content_type": "multipart/form-data",
                "parameters": [
                    {
                        "name": "text",
                        "type": "string",
                        "required": True,
                        "description": "Text to convert to speech"
                    },
                    {
                        "name": "hf_token",
                        "type": "string", 
                        "required": True,
                        "description": "HuggingFace API token"
                    }
                ],
                "response": {
                    "audio_url": "string - URL to the generated audio file"
                }
            },
            {
                "path": "/reset",
                "method": "POST", 
                "description": "Reset Cortana's memory and conversation history",
                "parameters": [],
                "response": {
                    "status": "string - success/error status",
                    "message": "string - confirmation message"
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Check API health status and uptime", 
                "parameters": [],
                "response": {
                    "status": "string - health status",
                    "uptime": "number - seconds since startup",
                    "version": "string - API version",
                    "service": "string - service name"
                }
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "Get comprehensive API documentation in JSON format",
                "parameters": [],
                "response": "object - This documentation structure"
            },
            {
                "path": "/api-docs", 
                "method": "GET",
                "description": "Get comprehensive API documentation in markdown format",
                "parameters": [],
                "response": {
                    "markdown": "string - Full API documentation in markdown format"
                }
            }
        ],
        "usage_notes": [
            "All file uploads must use multipart/form-data encoding",
            "API keys are required for most functionality", 
            "The chat endpoint supports multiple input types simultaneously",
            "Audio responses require a valid HuggingFace token",
            "File uploads (images, voice, document) are optional and can be used individually or together",
            "Multiple images can be uploaded per request, but only one voice and one document file"
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cortana AI Assistant API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h2 { color: #2c3e50; }
            h3 { color: #34495e; }
            code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
            ul { line-height: 1.6; }
        </style>
    </head>
    <body>
        <h2>Cortana AI Assistant API</h2>
        
        <p>A comprehensive API for interacting with Cortana AI Assistant with multi-modal capabilities.</p>
        
        <h3>Available Endpoints:</h3>
        
        <p><strong>GET Requests:</strong></p>
        <ul>
            <li><code>/health</code> - Check API health status and uptime</li>
            <li><code>/api-docs</code> - Get comprehensive API documentation</li>
            <li><code>/docs</code> - Get comprehensive API documentation in JSON format</li>
        </ul>
        
        <p><strong>POST Requests:</strong></p>
        <ul>
            <li><code>/chat</code> - Main chat endpoint with multi-modal support (text, multiple images, voice, document uploads)</li>
            <li><code>/text-to-speech</code> - Convert text to speech using HuggingFace TTS</li>
            <li><code>/reset</code> - Reset Cortana's memory and conversation history</li>
        </ul>
        
        <h3>Features:</h3>
        <ul>
            <li>Multi-modal input support (text, multiple images, voice, documents)</li>
            <li>Text-to-speech conversion</li>
            <li>Web search integration via Google and Tavily</li>
            <li>OpenAI GPT integration</li>
            <li>Composio tools integration</li>
            <li>Code execution integration</li>
            <li>Image search integration</li>
            <li>Memory management and conversation reset</li>
            <li>Health monitoring and uptime tracking</li>
        </ul>
    </body>
    </html>
    """
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

