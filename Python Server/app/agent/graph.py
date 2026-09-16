from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.agent.nodes import triage_node, kb_search_node, respond_node
from app.mcp.tools import MCP_TOOLS
import logging

def should_continue(state: AgentState):
    """Router function"""
    if state.get("next_action") == "tools":
        return "tools"
    return "kb_search"

# Initialize the ToolNode with our MCP mock tools
tools_node = ToolNode(MCP_TOOLS)

# Define the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("triage", triage_node)
workflow.add_node("tools", tools_node)
workflow.add_node("kb_search", kb_search_node)
workflow.add_node("respond", respond_node)

# Add Edges
workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    should_continue,
    {
        "tools": "tools",
        "kb_search": "kb_search"
    }
)

# After tools run, we might want to do a KB search or respond directly.
# For simplicity, we just route to respond so the LLM can summarize the tool output.
workflow.add_edge("tools", "respond")

# After KB search, we respond
workflow.add_edge("kb_search", "respond")

# End the graph after responding
workflow.add_edge("respond", END)

# Compile the Graph
app_graph = workflow.compile()
