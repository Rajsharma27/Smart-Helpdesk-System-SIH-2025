from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from app.models.schemas import ChatRequest, ChatResponse
from app.services.image_processing import process_image
from app.agent.graph import app_graph
from app.core.config import CHAT_SESSIONS_DIR
from app.core.guardrails import validate_input, redact_pii
import logging

router = APIRouter()

def get_session_history(session_id: str) -> FileChatMessageHistory:
    session_file = CHAT_SESSIONS_DIR / f"{session_id}.json"
    return FileChatMessageHistory(str(session_file))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 1. Guardrails (Input)
    is_valid, reason = validate_input(request.message)
    if not is_valid:
        return ChatResponse(responseText=reason)
        
    history = get_session_history(request.session_id)
    
    # 2. Image Processing (OCR)
    image_context = ""
    if request.image_data:
        image_context = process_image(request.image_data)
        
    # 3. Create Human Message
    user_message = HumanMessage(content=request.message)
    
    # 4. Invoke LangGraph
    initial_state = {
        "messages": history.messages + [user_message],
        "session_id": request.session_id,
        "image_context": image_context
    }
    
    try:
        logging.info(f"Invoking graph for session {request.session_id}")
        final_state = app_graph.invoke(initial_state)
        
        # The last message from the AI is a JSON string (model_dump_json of ChatResponse)
        final_ai_msg_str = final_state["messages"][-1].content
        
        # Parse it back to a ChatResponse Pydantic object
        import json
        response_data = json.loads(final_ai_msg_str)
        
        # 5. Guardrails (Output) - Safely redact text fields only
        redacted_response_text = redact_pii(response_data.get("responseText", ""))
        redacted_solution = None
        if response_data.get("solution"):
            redacted_solution = [redact_pii(step) for step in response_data["solution"]]
            
        final_response_obj = ChatResponse(
            responseText=redacted_response_text,
            solution=redacted_solution,
            ticket=response_data.get("ticket") # Ticket object remains intact for the frontend
        )
        
        # Update Memory
        history.add_message(user_message)
        # Save the redacted string in memory so we don't leak PII in future context
        history.add_ai_message(AIMessage(content=final_response_obj.model_dump_json()))
        
        return final_response_obj
        
    except Exception as e:
        logging.error(f"Error in graph execution: {e}", exc_info=True)
        return ChatResponse(responseText="An error occurred while processing your request.")

@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        messages.append({"type": role, "content": msg.content})
    return {"history": messages}
