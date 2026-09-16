from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agent.state import AgentState
from app.agent.llm import llm, llm_with_tools
from app.services.kb_retriever import retrieve_knowledge
from app.models.schemas import ChatResponse, Ticket, AIAnalysis
import logging

def triage_node(state: AgentState) -> dict:
    """
    Decides whether the query needs more info, can be answered directly, or needs a tool.
    """
    logging.info("--- NODE: TRIAGE ---")
    messages = state.get("messages", [])
    
    # We add the image context if it exists to the latest message
    latest_msg = messages[-1].content
    if state.get("image_context"):
        latest_msg = f"[Image OCR Context: {state['image_context']}]\n{latest_msg}"
        
    sys_msg = SystemMessage(content="""
    You are an IT Helpdesk Triage Agent.
    Your job is to read the user's request and decide the next step.
    If the request is too vague (e.g. 'help me', 'broken'), ask for clarification.
    Otherwise, respond normally and use tools if necessary.
    """)
    
    # We pass the messages to the LLM bound with tools
    # If the LLM decides to use a tool, it will return an AIMessage with tool_calls
    response = llm_with_tools.invoke([sys_msg] + messages)
    
    # If the LLM made a tool call, we route to the tools node
    if hasattr(response, 'tool_calls') and len(response.tool_calls) > 0:
        return {"messages": [response], "next_action": "tools"}
        
    return {"messages": [response], "next_action": "respond"}

def kb_search_node(state: AgentState) -> dict:
    """
    Retrieves context from the Knowledge Base before the final response.
    """
    logging.info("--- NODE: KB SEARCH ---")
    latest_human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    kb_context = retrieve_knowledge(latest_human_msg)
    return {"kb_context": kb_context}

def respond_node(state: AgentState) -> dict:
    """
    Generates the final structured response (ChatResponse).
    """
    logging.info("--- NODE: RESPOND ---")
    messages = state["messages"]
    kb_context = state.get("kb_context", "")
    
    sys_msg = SystemMessage(content=f"""
    You are an IT Helpdesk Assistant. Answer the user's query.
    
    KNOWLEDGE BASE CONTEXT:
    {kb_context}
    
    INSTRUCTIONS:
    1. If the knowledge base context helps, use it to provide a 'solution'.
    2. If tools were run previously, use their output to formulate your answer.
    3. IMPORTANT: If the user's issue is complex (e.g. hardware failure), or they explicitly ask to raise a ticket, or if tools fail to resolve the issue, you MUST generate a 'ticket' object.
       - The 'ticket.source' should be "Chatbot".
       - The 'ticket.status' should be "Open".
    4. If the issue is simple and can be resolved directly, provide the steps in the 'solution' array and leave 'ticket' null.
    5. Always provide a friendly 'responseText' summarizing your action to the user.
    """)
    
    # Use with_structured_output to force the LLM to output the exact Pydantic schema
    structured_llm = llm.with_structured_output(ChatResponse)
    
    # We pass the sys_msg plus the conversation history
    response: ChatResponse = structured_llm.invoke([sys_msg] + messages)
    
    # The output is no longer a raw AIMessage, but a parsed Pydantic object.
    # To keep the state clean for the graph (which expects messages), we convert it back to a message representation.
    # Alternatively, we could just return it as a final state attribute.
    # We will attach the raw json string as an AIMessage so the graph is happy.
    
    final_msg = AIMessage(content=response.model_dump_json())
    
    return {"messages": [final_msg]}
