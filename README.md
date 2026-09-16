# Smart Helpdesk AI System (SIH 2025)

An agentic, multi-service IT helpdesk platform that combines a **Next.js** frontend, a **Node.js/Express** database API backend, and an **Agentic Python (FastAPI + LangGraph)** AI engine for intelligent IT troubleshooting, knowledge-base retrieval (RAG), tool execution (MCP), and automated ticket creation.

---

## 🏗 System Architecture

```text
[ Next.js Frontend ]
        │
        ├── (REST API: /auth, /tickets) ──────────► [ Node.js Express Backend ] ──► [ Database ]
        │
        └── (REST API: /chat with OCR support) ──► [ Python AI Server (FastAPI) ]
                                                            │
                                                            ▼
                                                   [ LangGraph State Machine ]
                                                      ├── Triage Node
                                                      ├── MCP Tool Execution (Password reset, server check)
                                                      ├── RAG KB Retrieval (ChromaDB)
                                                      └── Guardrails & Structured Output (ChatResponse)
```

---

## 🚀 Key Features

* **Agentic LangGraph Workflow:** Replaces monolithic prompts with a modular state graph (Triage -> Tool Calling -> KB Search -> Structured Response).
* **Model Context Protocol (MCP) Tools:** Safely executes IT administrative tools (e.g., password resets, server health checks, ticket creation) directly within the graph execution cycle.
* **Grounding via RAG (ChromaDB):** Answers user questions using vector search over company IT manuals and policies to prevent LLM hallucinations.
* **Security & Guardrails:** Custom Pydantic validation and regex filters automatically redact PII (passwords, SSNs) and block malicious/off-topic inputs before processing.
* **Screenshot OCR Processing:** Extracts error text from uploaded images/screenshots using PyTesseract and integrates OCR results into the reasoning loop.
* **Seamless Full-Stack Integration:** Returns structured JSON (`ChatResponse` with `ticket` payload) that the frontend receives to automatically trigger database ticket generation in the Node.js backend.
* **AIOps & Observability:** Integrated with LangSmith tracing to monitor node transitions, execution latencies, and LLM calls in real time.

---

## 📁 Repository Structure

```text
Smart-Helpdesk-System-SIH-2025/
├── frontend/                 # Next.js Chatbot UI & Support Portal
│   ├── src/
│   │   ├── components/       # Chatbot UI, Buttons, Input boxes
│   │   ├── services/         # API clients for Node backend & Python server
│   │   └── app/              # Next.js Pages & Layouts
├── BackEnd/                  # Node.js + Express API Backend
│   ├── src/
│   │   ├── controllers/     # Ticket & User management logic
│   │   ├── models/          # Database schemas
│   │   └── routes/          # Express route definitions
├── Python Server/            # FastAPI + LangGraph AI Core
│   ├── app/
│   │   ├── agent/           # LangGraph state graph, nodes, and LLM setup
│   │   ├── api/             # FastAPI /chat endpoints & memory handling
│   │   ├── core/            # Config, AIOps, and Guardrails logic
│   │   ├── mcp/             # MCP tools (Password Reset, Ticket Creation, etc.)
│   │   ├── models/          # Pydantic schemas (ChatResponse, Ticket, etc.)
│   │   └── services/        # ChromaDB vector store retriever & Image OCR
│   └── main.py              # Python FastAPI entrypoint
└── README.md
```

---

## 🛠 Tech Stack

- **Frontend:** Next.js, React, TailwindCSS, Axios, Lucide Icons
- **Backend:** Node.js, Express, MongoDB/PostgreSQL
- **AI Core:** Python 3.10+, FastAPI, LangGraph, LangChain Core, Gemini 2.5 Flash, ChromaDB, PyTesseract, Pydantic
- **Observability:** LangSmith

---

## 🚦 Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.10+)
- Tesseract OCR installed on your system
- Google Gemini API Key

---

### 1. Python AI Server Setup

```bash
cd "Python Server"

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file:
# GOOGLE_API_KEY="your_gemini_api_key"
# LANGCHAIN_API_KEY="your_langsmith_key"  # Optional for tracing

# Run the server
fastapi dev main.py
```
*The Python AI server will start on `http://localhost:8000`.*

---

### 2. Node.js Backend Setup

```bash
cd BackEnd

# Install dependencies
npm install

# Run backend
npm start
```

---

### 3. Next.js Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run frontend dev server
npm run dev
```
*Access the web app at `http://localhost:3000`.*

---

## 🛡 Guardrails & Privacy

- **Input Guardrail:** Filters malicious prompts, prompt injection attempts, and non-IT queries.
- **Output Guardrail:** Redacts sensitive credentials (e.g., passwords, SSNs) from text outputs before returning responses to the user or storing chat history.
