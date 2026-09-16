from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import GOOGLE_API_KEY
from app.mcp.tools import MCP_TOOLS

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# Bind the tools to the LLM so it knows what actions it can take
llm_with_tools = llm.bind_tools(MCP_TOOLS)