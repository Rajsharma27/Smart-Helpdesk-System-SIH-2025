from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state for the Helpdesk LangGraph.
    """
    # The list of messages in the conversation. `add_messages` appends new messages.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Session ID for tracing and memory
    session_id: str
    
    # Any context retrieved from the Knowledge Base
    kb_context: Optional[str]
    
    # Information parsed from image OCR if present
    image_context: Optional[str]
    
    # Internal routing flag
    next_action: str
