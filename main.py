import json
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

import config
from image_processing import process_image
from llm_setup import main_chain, get_session_history
from schemas import ChatRequest, ChatResponse


# ----------------- FastAPI App Initialization -----------------
app = FastAPI(
    title="Smart Helpdesk AI Prototype",
    description="An AI-powered IT Helpdesk with a schema-aware, merged-context chain",
    version="7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Core Logic -----------------
def process_query_ai(query: str, session_id: str, image_data: Optional[str] = None) -> dict:
    fallback_response = {"solution": None, "ticket": None, "responseText": "I'm sorry, I'm having trouble connecting."}
    try:
        history = get_session_history(session_id)
        effective_query = query or ""

        # Use OCR tool if image is attached
        if image_data:
            extracted_text = process_image(image_data)
            effective_query = f"SCREENSHOT OCR RESULT:\n{extracted_text}\n\nUSER MESSAGE: \"{query}\""

        result = main_chain.invoke({"chat_history": history.messages, "query": effective_query})
        
        if image_data:
            user_message_content = [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_message_content = [{"type": "text", "text": query}]
            
        user_message_for_history = HumanMessage(content=user_message_content)
        
        content = result.content.strip().removeprefix("```json").removesuffix("```").strip()
        
        history.add_message(user_message_for_history)
        history.add_ai_message(AIMessage(content=content))
        
        response_data = json.loads(content)
        
        logging.info(f"AI Response for session {session_id}: {json.dumps(response_data)}")
        return response_data
    except Exception as e:
        raw_output = result.content if "result" in locals() else "N/A"
        logging.error(
            f"Error processing session {session_id}. Raw LLM output: '{raw_output}'",
            exc_info=True
        )
        return fallback_response


# ----------------- API Endpoints -----------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return process_query_ai(request.message, request.session_id, request.image_data)

@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            content_data = []
            content_list = msg.content if isinstance(msg.content, list) else [{"type": "text", "text": msg.content}]
            
            for part in content_list:
                if part.get("type") == "text":
                    content_data.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    content_data.append({"type": "image_url", "url": url})
            messages.append({"type": "human", "content": content_data})

        elif isinstance(msg, AIMessage):
            try:
                cleaned_content = msg.content.strip().removeprefix("```json").removesuffix("```").strip()
                parsed_content = json.loads(cleaned_content)
                messages.append({"type": "ai", "content": parsed_content})
            except json.JSONDecodeError:
                messages.append({"type": "ai", "content": {"responseText": "Error: Could not load this message."}})
                
    return {"history": messages}

@app.get("/")
def root():
    return {"message": "Helpdesk AI (Gemini) Chat is running 🚀"}


# ----------------- FastAPI App Initialization -----------------
app = FastAPI(
    title="Smart Helpdesk AI Prototype",
    description="An AI-powered IT Helpdesk with a schema-aware, merged-context chain",
    version="7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Core Logic -----------------
def process_query_ai(query: str, session_id: str, image_data: Optional[str] = None) -> dict:
    fallback_response = {"solution": None, "ticket": None, "responseText": "I'm sorry, I'm having trouble connecting."}
    try:
        history = get_session_history(session_id)
        effective_query = query or ""

        # Use OCR tool if image is attached
        if image_data:
            extracted_text = process_image(image_data)
            effective_query = f"SCREENSHOT OCR RESULT:\n{extracted_text}\n\nUSER MESSAGE: \"{query}\""

        result = main_chain.invoke({"chat_history": history.messages, "query": effective_query})
        
        if image_data:
            user_message_content = [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_message_content = [{"type": "text", "text": query}]
            
        user_message_for_history = HumanMessage(content=user_message_content)
        
        content = result.content.strip().removeprefix("```json").removesuffix("```").strip()
        
        history.add_message(user_message_for_history)
        history.add_ai_message(AIMessage(content=content))
        
        response_data = json.loads(content)
        
        logging.info(f"AI Response for session {session_id}: {json.dumps(response_data)}")
        return response_data
    except Exception as e:
        raw_output = result.content if "result" in locals() else "N/A"
        logging.error(
            f"Error processing session {session_id}. Raw LLM output: '{raw_output}'",
            exc_info=True
        )
        return fallback_response


# ----------------- API Endpoints -----------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return process_query_ai(request.message, request.session_id, request.image_data)

@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            content_data = []
            content_list = msg.content if isinstance(msg.content, list) else [{"type": "text", "text": msg.content}]
            
            for part in content_list:
                if part.get("type") == "text":
                    content_data.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    content_data.append({"type": "image_url", "url": url})
            messages.append({"type": "human", "content": content_data})

        elif isinstance(msg, AIMessage):
            try:
                cleaned_content = msg.content.strip().removeprefix("```json").removesuffix("```").strip()
                parsed_content = json.loads(cleaned_content)
                messages.append({"type": "ai", "content": parsed_content})
            except json.JSONDecodeError:
                messages.append({"type": "ai", "content": {"responseText": "Error: Could not load this message."}})
                
    return {"history": messages}

@app.get("/")
def root():
    return {"message": "Helpdesk AI (Gemini) Chat is running 🚀"}


# ----------------- FastAPI App Initialization -----------------
app = FastAPI(
    title="Smart Helpdesk AI Prototype",
    description="An AI-powered IT Helpdesk with a schema-aware, merged-context chain",
    version="7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Core Logic -----------------
def process_query_ai(query: str, session_id: str, image_data: Optional[str] = None) -> dict:
    fallback_response = {"solution": None, "ticket": None, "responseText": "I'm sorry, I'm having trouble connecting."}
    try:
        history = get_session_history(session_id)
        effective_query = query or ""

        # Use OCR tool if image is attached
        if image_data:
            extracted_text = process_image(image_data)
            effective_query = f"SCREENSHOT OCR RESULT:\n{extracted_text}\n\nUSER MESSAGE: \"{query}\""

        result = main_chain.invoke({"chat_history": history.messages, "query": effective_query})
        
        if image_data:
            user_message_content = [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_message_content = [{"type": "text", "text": query}]
            
        user_message_for_history = HumanMessage(content=user_message_content)
        
        content = result.content.strip().removeprefix("```json").removesuffix("```").strip()
        
        history.add_message(user_message_for_history)
        history.add_ai_message(AIMessage(content=content))
        
        response_data = json.loads(content)
        
        logging.info(f"AI Response for session {session_id}: {json.dumps(response_data)}")
        return response_data
    except Exception as e:
        raw_output = result.content if "result" in locals() else "N/A"
        logging.error(
            f"Error processing session {session_id}. Raw LLM output: '{raw_output}'",
            exc_info=True
        )
        return fallback_response


# ----------------- API Endpoints -----------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return process_query_ai(request.message, request.session_id, request.image_data)

@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            content_data = []
            content_list = msg.content if isinstance(msg.content, list) else [{"type": "text", "text": msg.content}]
            
            for part in content_list:
                if part.get("type") == "text":
                    content_data.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    content_data.append({"type": "image_url", "url": url})
            messages.append({"type": "human", "content": content_data})

        elif isinstance(msg, AIMessage):
            try:
                cleaned_content = msg.content.strip().removeprefix("```json").removesuffix("```").strip()
                parsed_content = json.loads(cleaned_content)
                messages.append({"type": "ai", "content": parsed_content})
            except json.JSONDecodeError:
                messages.append({"type": "ai", "content": {"responseText": "Error: Could not load this message."}})
                
    return {"history": messages}

@app.get("/")
def root():

    return {"message": "Helpdesk AI (Gemini) Chat is running 🚀"}

