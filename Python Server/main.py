from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as chat_router

# ----------------- FastAPI App Initialization -----------------
app = FastAPI(
    title="Smart Helpdesk AI (Agentic Edition)",
    description="An AI-powered IT Helpdesk with LangGraph, MCP, and Guardrails",
    version="8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Routing -----------------
app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "Agentic Helpdesk AI is running 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
