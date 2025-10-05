from typing import Optional, List
from pydantic import BaseModel

class AIAnalysis(BaseModel):
    sentiment: str
    keywords: List[str]

class Ticket(BaseModel):
    title: str
    description: str
    priority: str
    category: str
    subcategory: str
    status: str
    source: str
    tags: List[str]
    aiAnalysis: AIAnalysis

class ChatResponse(BaseModel):
    solution: Optional[List[str]] = None
    ticket: Optional[Ticket] = None
    responseText: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    image_data: Optional[str] = None
