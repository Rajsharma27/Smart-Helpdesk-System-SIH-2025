from langchain_core.tools import tool
import logging

@tool
def reset_password(username: str) -> str:
    """
    Resets the password for a given user.
    Use this tool ONLY when a user explicitly asks to reset their password or login credentials.
    """
    logging.info(f"MCP Tool Execution: reset_password for {username}")
    # Simulate API call to Active Directory / Identity Provider
    return f"Successfully reset password for user {username}. A temporary password has been emailed to them."

@tool
def check_server_status(server_name: str) -> str:
    """
    Checks the current status of a server or service (e.g., VPN, Email server, Database).
    """
    logging.info(f"MCP Tool Execution: check_server_status for {server_name}")
    # Simulate ping / health check
    if "vpn" in server_name.lower():
        return f"Server {server_name} is currently experiencing degraded performance. ETA for fix is 2 hours."
    return f"Server {server_name} is ONLINE and healthy."

@tool
def create_jira_ticket(title: str, description: str, priority: str = "Medium") -> str:
    """
    Creates an IT support ticket in the ticketing system (e.g., Jira/ServiceNow) for complex issues
    that cannot be solved automatically.
    """
    logging.info(f"MCP Tool Execution: create_jira_ticket with title '{title}'")
    # Simulate ticket creation
    ticket_id = "IT-8842"
    return f"Successfully created ticket {ticket_id}. Priority: {priority}."

# List of all available tools to bind to the LLM
MCP_TOOLS = [reset_password, check_server_status, create_jira_ticket]
