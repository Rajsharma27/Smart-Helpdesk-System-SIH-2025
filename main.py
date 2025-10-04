import os
import json
import logging
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

# ----------------- Basic Configuration -----------------
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not GOOGLE_API_KEY:
    logging.error("FATAL ERROR: GOOGLE_API_KEY environment variable not found.")
    logging.error(f"Please ensure a .env file exists at {dotenv_path} and contains GOOGLE_API_KEY='your_api_key'")
    sys.exit(1)

# ----------------- FastAPI App Initialization -----------------
app = FastAPI(
    title="Smart Helpdesk AI Prototype",
    description="An AI-powered IT Helpdesk with a schema-aware, merged-context chain",
    version="6.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- LangChain LLM and Memory Setup -----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ----------------- Prompt Engineering -----------------

vision_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert at analyzing images. Describe the image and extract the exact text of any error messages you see."),
        ("human", "{vision_query}"),
    ]
)

# ...existing code...
main_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a schema-aware AI IT Helpdesk assistant. Your primary goal is to gather sufficient information before taking action.

**Your Reasoning Process:**
1.  **Synthesize Context:** Review the `chat_history` and the user's latest `query`.
2.  **INFORMATION SUFFICIENCY CHECK:**
    * **Is the query vague?** If the user's request is too generic to be actionable (e.g., "hardware issue," "my computer is slow," "it's not working"), you MUST ask one or two clarifying questions. Do not proceed further. Your response should contain a `responseText` asking for more details, and both `solution` and `ticket` fields MUST be `null`.
    * **Is the information sufficient?** If you have a specific error message or a clear description of the problem (e.g., "my CPU is not working," or from a screenshot analysis), proceed to the next step.
3.  **DECISION GATE:** (Only if you have sufficient information)
    * **Path A - Simple Issue:** For simple, solvable problems ('incorrect password', 'login failed'), provide a `solution`. The `ticket` field must be `null`.
    * **Path B - Complex Issue or Escalation:** For complex problems (hardware failure, server errors) or if the user has confirmed a solution failed, generate a `ticket`. The `solution` field must be `null`.
4.  **Populate Schema:** Based on your decision, populate the JSON fields.
5.  **Final Response:** Your entire output MUST BE the JSON object and nothing else.

**JSON Response Format and Ticket Schema:**
{{{{
    "solution": ["Step 1..."] or null,
    "ticket": {{{{
        "title": "string",
        "description": "string",
        "priority": "Low/Medium/High",
        "category": "Password Reset/Hardware/Software/Network/Other",
        "subcategory": "string",
        "status": "Open",
        "source": "Chatbot",
        "userId": "string",
        "username": "string",
        "tags": ["string"],
        "aiAnalysis": {{{{
            "sentiment": "positive/neutral/negative",
            "keywords": ["string"]
        }}}}
    }}}} or null,
    "responseText": "A conversational summary of your action or your clarifying questions."
}}}}

**Important Notes:**
*   The `status` field in the `ticket` object should initially be set to "Open" when the ticket is created by the chatbot.
*   The `source` field should be set to "Chatbot".
*   The `aiAnalysis` field provides sentiment analysis and keywords extracted from the user's input.
""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])
# ...existing code...

# ----------------- LangChain Core Runnable Chains -----------------
vision_chain = vision_prompt | llm
main_chain = main_prompt | llm

# ----------------- Pydantic Schemas -----------------
class ChatRequest(BaseModel):
    message: str
    session_id: str
    image_data: Optional[str] = None

# ----------------- Core Logic -----------------
def process_query_ai(query: str, session_id: str, image_data: Optional[str] = None) -> dict:
    fallback_response = {"solution": None, "ticket": None, "responseText": "I'm sorry, I'm having trouble connecting."}
    try:
        history = get_session_history(session_id)
        effective_query = query or ("Analyze the attached screenshot." if image_data else "")

        final_query = effective_query
        if image_data:
            vision_message = HumanMessage(content=[{"type": "text", "text": "Analyze screenshot."}, {"type": "image_url", "image_url": {"url": image_data}}])
            analysis_result = vision_chain.invoke({"vision_query": [vision_message]})
            image_analysis = analysis_result.content
            final_query = f"SCREENSHOT ANALYSIS:\n{image_analysis}\n\nUSER'S MESSAGE: \"{effective_query}\""

        result = main_chain.invoke({"chat_history": history.messages, "query": final_query})
        
        user_message_for_history = HumanMessage(content=[{"type": "text", "text": query}] + ([{"type": "image_url", "image_url": {"url": image_data}}] if image_data else []))
        history.add_message(user_message_for_history)
        history.add_ai_message(AIMessage(content=result.content))
        
        content = result.content.strip().removeprefix("```json").removesuffix("```").strip()
        response_data = json.loads(content)

        print(json.dumps(response_data, indent=2))
        
        return response_data
    except Exception as e:
        logging.error(f"Error in process_query_ai for session {session_id}: {e}", exc_info=True)
        return fallback_response

# ----------------- API Endpoints -----------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return process_query_ai(request.message, request.session_id, request.image_data)

@app.get("/")
def root():
    return {"message": "Helpdesk AI (Gemini) Chat is running 🚀"}



