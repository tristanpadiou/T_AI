from cortana_agent import Cortana_agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import time

app = FastAPI(title="Cortana API", description="API for interacting with Cortana AI Assistant")

class QueryRequest(BaseModel):
    query: str

class InitRequest(BaseModel):
    api_keys: Dict[str, str]

# Initialize Cortana agent instance
cortana = None
startup_time = time.time()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "uptime": time.time() - startup_time,
        "version": "0.1.0",
        "service": "Cortana API",
        "cortana_initialized": cortana is not None
    }

@app.post("/initialize")
async def initialize_cortana(request: InitRequest):
    global cortana
    try:
        cortana = Cortana_agent(request.api_keys)
        return {"status": "success", "message": "Cortana initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: QueryRequest):
    global cortana
    
    if cortana is None:
        raise HTTPException(
            status_code=400,
            detail="Cortana not initialized. Please call /initialize endpoint first."
        )
    
    try:
        response = await cortana.chat(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

