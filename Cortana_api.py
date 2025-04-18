from cortana_agent import Cortana_agent
from fastapi import FastAPI, HTTPException, Form, Header
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import time
import json
from google.auth.credentials import Credentials

app = FastAPI(title="Cortana API", description="API for interacting with Cortana AI Assistant")

# Initialize Cortana agent instance
cortana = Cortana_agent()
startup_time = time.time()

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
    x_google_credentials: Optional[str] = Header(None)
):
    try:
        api_keys = {
            "google_api_key": google_api_key,
            "tavily_key": tavily_key
        }
        if pse:
            api_keys["pse"] = pse
        if x_google_credentials:
            creds_data = json.loads(x_google_credentials)
            creds = Credentials.from_authorized_user_info(
                info=creds_data.get('info'),
                scopes=creds_data.get('scopes')
            )
            api_keys["creds"] = creds
            
        response = await cortana.chat(query, api_keys)
        return {"response": response}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid credentials format in X-Google-Credentials header"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_cortana():
    try:
        cortana.reset_memory()
        return {"status": "success", "message": "Cortana memory reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

