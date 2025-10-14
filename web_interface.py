"""
FastAPI Web Interface for Healthcare AI Agent
Provides REST API for web/mobile integration
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime
import uuid
from healthcare_agent_main import HealthcareAgent
from healthcare_config import Config
import os
from typing import Any
from modules.rag_system import MedicalRAGSystem   # <- save your RAG class into rag_system.py
from langchain_groq import ChatGroq


# Initialize FastAPI
app = FastAPI(
    title="Healthcare AI Agent API",
    description="AI-powered healthcare decision support for PHC workers",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static frontend (index.html) and assets
if os.path.isdir("public"):
    app.mount("/assets", StaticFiles(directory="public"), name="assets")

    @app.get("/{full_path:path}")
    async def static_catch_all(full_path: str):
        # Let API routes take precedence; this catch-all only serves the SPA
        if full_path.startswith("api/") or full_path in {"health", ""}:
            # fall through to API routes (they are already defined)
            pass
        return FileResponse("public/index.html")

# Initialize agent
config = Config()
agent = HealthcareAgent()

# --- RAG system bootstrap ---
# Use env var for your Groq key (DO NOT hardcode keys)

rag_llm = ChatGroq(
    model=config.GROQ_MODEL,  # or a specific one, e.g., "llama-3.1-8b-instant"
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

rag_system = MedicalRAGSystem(
    llm=rag_llm,
    persist_directory="modules/vector_store"      # adjust if you want a different path
)

# optionally attach onto agent so other code can use it
agent.rag_system = rag_system

# Store sessions (in production, use Redis or database)
sessions = {}


# Pydantic Models
class VitalSigns(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None


class PatientData(BaseModel):
    id: Optional[str] = None
    age: int
    gender: str
    chief_complaint: str
    symptoms: List[str] = []
    vital_signs: Optional[VitalSigns] = None
    medical_history: List[str] = []


class TriageRequest(BaseModel):
    patient: PatientData
    language: str = "English"
    additional_notes: Optional[str] = None


class ReferralRequest(BaseModel):
    patient: PatientData
    facility_capabilities: Dict = {}
    triage_result: Optional[Dict] = None
    language: str = "English"


class ResourceRequest(BaseModel):
    facility_data: Dict
    current_demand: Dict = {}
    language: str = "English"


class RoutingRequest(BaseModel):
    patient: PatientData
    facility_context: Dict = {}
    language: str = "English"


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    patient_data: Optional[PatientData] = None
    facility_data: Optional[Dict] = None
    language: str = "English"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: datetime
    sources: List[str] = []


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "status": "online",
        "service": "Healthcare AI Agent",
        "version": "1.0.0",
        "endpoints": {
            "triage": "/api/triage",
            "referral": "/api/referral",
            "resources": "/api/resources",
            "routing": "/api/routing",
            "chat": "/api/chat"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": config.GROQ_MODEL,
        "languages_supported": config.SUPPORTED_LANGUAGES
    }


# Triage endpoint
@app.post("/api/triage")
async def perform_triage(request: TriageRequest):
    """
    Perform AI-assisted triage assessment
    
    Returns priority level, reasoning, and immediate actions
    """
    try:
        # Convert Pydantic model to dict
        patient_dict = request.patient.dict()
        
        # If vital signs provided, convert nested model
        if request.patient.vital_signs:
            patient_dict['vital_signs'] = request.patient.vital_signs.dict(exclude_none=True)
        
        # Run agent
        result = agent.run(
            user_input=f"Perform triage assessment. {request.additional_notes or ''}",
            patient_data=patient_dict,
            language=request.language
        )
        
        # Extract response
        response_text = ""
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                response_text += msg.content + "\n"
        
        return {
            "success": True,
            "assessment": response_text,
            "patient_id": request.patient.id or str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "language": request.language
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Triage error: {str(e)}")


# Referral endpoint
@app.post("/api/referral")
async def generate_referral(request: ReferralRequest):
    """
    Generate referral recommendation
    
    Returns referral decision, specialty, and guidance
    """
    try:
        patient_dict = request.patient.dict()
        if request.patient.vital_signs:
            patient_dict['vital_signs'] = request.patient.vital_signs.dict(exclude_none=True)
        
        result = agent.run(
            user_input="Generate referral recommendation for this patient",
            patient_data=patient_dict,
            facility_data=request.facility_capabilities,
            additional_context={"triage": request.triage_result} if request.triage_result else {},
            language=request.language
        )
        
        response_text = ""
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                response_text += msg.content + "\n"
        
        return {
            "success": True,
            "recommendation": response_text,
            "patient_id": request.patient.id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Referral error: {str(e)}")


# Resource allocation endpoint
@app.post("/api/resources")
async def optimize_resources(request: ResourceRequest):
    """
    Optimize resource allocation
    
    Returns recommendations for bed management, staff deployment, etc.
    """
    try:
        result = agent.run(
            user_input="Optimize resource allocation for current facility status",
            facility_data=request.facility_data,
            additional_context={"demand": request.current_demand},
            language=request.language
        )
        
        response_text = ""
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                response_text += msg.content + "\n"
        
        return {
            "success": True,
            "recommendations": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource optimization error: {str(e)}")


# Patient routing endpoint
@app.post("/api/routing")
async def route_patient(request: RoutingRequest):
    """
    Determine optimal patient routing
    
    Returns routing plan and estimated times
    """
    try:
        patient_dict = request.patient.dict()
        if request.patient.vital_signs:
            patient_dict['vital_signs'] = request.patient.vital_signs.dict(exclude_none=True)
        
        result = agent.run(
            user_input="Determine optimal routing for this patient",
            patient_data=patient_dict,
            facility_data=request.facility_context,
            language=request.language
        )
        
        response_text = ""
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                response_text += msg.content + "\n"
        
        return {
            "success": True,
            "routing_plan": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")


# Chat endpoint for conversational interaction
import traceback
import sys
from datetime import datetime
import logging

# Setup logging at the top of your file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational interface with the AI agent
    Maintains session context for multi-turn conversation
    """
    try:
        # 1) Session
        session_id = request.session_id or str(uuid.uuid4())
        session = sessions.get(session_id, {"history": []})

        # 2) Prepare optional data
        patient_dict = request.patient_data.dict() if request.patient_data else None
        if patient_dict and request.patient_data.vital_signs:
            patient_dict["vital_signs"] = request.patient_data.vital_signs.dict(exclude_none=True)

        # 3) Build compact context
        MAX_TURNS = 5
        recent = session["history"][-MAX_TURNS:]
        user_only_context = "\n".join([f"User: {h['user']}" for h in recent if h.get("user")])

        # NEW: pull RAG context for the current message
        rag_hits = rag_system.retrieve(request.message, k=4)
        rag_context = "\n\n".join(
            [f"Source {i+1}: {h['content']}" for i, h in enumerate(rag_hits)]
        )

        # Compose final prompt
        user_prompt = f"""
You are a helpful, concise clinical assistant for a PHC worker.
Use clear, structured answers with brief bullet points where appropriate.
If uncertain, ask a brief clarifying question and suggest safe next steps.
Conversation so far (user turns only):
{user_only_context}

Relevant knowledge (do not quote verbatim; synthesize; cite like 'Source 1'):
{rag_context or 'No retrieved sources.'}

New user message:
{request.message}
""".strip()

        # 4) Call agent
        result = agent.run(
            user_input=user_prompt,
            patient_data=patient_dict,
            facility_data=request.facility_data,
            language=request.language,
        )

        # 5) Extract ONLY the final assistant message
        response_text = ""
        messages = result.get("messages", [])
        if messages:
            last = next((m for m in reversed(messages) if hasattr(m, "content") and m.content), None)
            response_text = (last.content if last else "")
        else:
            response_text = str(result.get("answer") or result.get("output") or "")

        # Optional echo guard
        if session["history"]:
            prev_assistant = session["history"][-1].get("assistant", "")
            if prev_assistant and response_text.endswith(prev_assistant):
                response_text = response_text[: -len(prev_assistant)].rstrip()

        # Sources if provided
        sources = []
        if "retrieved_docs" in result:
            sources = [doc.get("source", "Unknown") for doc in result["retrieved_docs"]]

        # 6) Update memory
        session["history"].append({
            "user": request.message,
            "assistant": response_text.strip(),
        })
        if len(session["history"]) > 50:
            session["history"] = session["history"][-50:]

        session["last_interaction"] = datetime.now()
        session["message_count"] = session.get("message_count", 0) + 1
        session["language"] = request.language
        sessions[session_id] = session

        return ChatResponse(
            session_id=session_id,
            response=response_text.strip(),
            timestamp=datetime.now(),
            sources=sources,
        )

    except Exception as e:
        # ============================================
        # ENHANCED ERROR HANDLING WITH FULL TRACEBACK
        # ============================================
        
        # Get full traceback information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Format the full traceback as a string
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        full_traceback = ''.join(tb_lines)
        
        # Extract specific error location
        tb_list = traceback.extract_tb(exc_traceback)
        if tb_list:
            last_frame = tb_list[-1]
            error_file = last_frame.filename
            error_line = last_frame.lineno
            error_function = last_frame.name
            error_code = last_frame.line
        else:
            error_file = "Unknown"
            error_line = "Unknown"
            error_function = "Unknown"
            error_code = "Unknown"
        
        # Log the full error with traceback
        logger.error(
            f"Chat endpoint error:\n"
            f"Type: {exc_type.__name__}\n"
            f"Message: {str(exc_value)}\n"
            f"File: {error_file}\n"
            f"Line: {error_line}\n"
            f"Function: {error_function}\n"
            f"Code: {error_code}\n"
            f"Full Traceback:\n{full_traceback}"
        )
        
        # Option 1: Return detailed error (DEVELOPMENT ONLY - not for production)
        error_detail = {
            "error_type": exc_type.__name__,
            "error_message": str(exc_value),
            "file": error_file,
            "line": error_line,
            "function": error_function,
            "code_line": error_code,
            "full_traceback": full_traceback,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Option 2: Return sanitized error (PRODUCTION)
        # Uncomment for production to hide sensitive details
        # error_detail = f"Chat error: {str(e)}"
        
        raise HTTPException(
            status_code=500,
            detail=error_detail  # Use error_detail for dev, str(e) for prod
        )


# Alternative: Dedicated error logging function
# def log_detailed_error(e: Exception, context: dict = None):
#     """
#     Utility function to log detailed error information
    
#     Args:
#         e: The exception object
#         context: Additional context (request data, session info, etc.)
#     """
#     exc_type, exc_value, exc_traceback = sys.exc_info()
    
#     # Get traceback details
#     tb_list = traceback.extract_tb(exc_traceback)
    
#     error_info = {
#         "timestamp": datetime.now().isoformat(),
#         "error_type": exc_type.__name__ if exc_type else "Unknown",
#         "error_message": str(exc_value),
#         "traceback": []
#     }
    
#     # Build traceback list
#     for frame in tb_list:
#         error_info["traceback"].append({
#             "file": frame.filename,
#             "line": frame.lineno,
#             "function": frame.name,
#             "code": frame.line
#         })
    
#     # Add context if provided
#     if context:
#         error_info["context"] = context
    
#     # Log as JSON for easy parsing
#     logger.error(f"Detailed error: {error_info}")
    
#     return error_info


# Usage with the alternative function:
"""
except Exception as e:
    error_details = log_detailed_error(e, context={
        "session_id": request.session_id,
        "message": request.message[:100],  # First 100 chars
        "has_patient_data": request.patient_data is not None
    })
    
    raise HTTPException(
        status_code=500,
        detail=f"Chat error at {error_details['traceback'][-1]['file']}:"
               f"{error_details['traceback'][-1]['line']} - {str(e)}"
    )
"""

# Get supported languages
@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return {
        "languages": config.SUPPORTED_LANGUAGES,
        "default": "English"
    }


# Medical knowledge search
@app.post("/api/knowledge/search")
async def search_knowledge(query: str, category: Optional[str] = None, language: str = "English"):
    """
    Search medical knowledge base
    
    Returns relevant information from RAG system
    """
    try:
        result = agent.rag_system.answer_with_sources(query)
        
        return {
            "success": True,
            "query": query,
            "answer": result['answer'],
            "sources": [
                {
                    "content": doc['content'][:200] + "...",
                    "source": doc['source']
                }
                for doc in result['sources']
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# Emergency protocols
@app.get("/api/protocols/emergency/{condition}")
async def get_emergency_protocol(condition: str, language: str = "English"):
    """Get emergency protocol for specific condition"""
    try:
        result = agent.rag_system.get_emergency_protocol(condition)
        
        return {
            "success": True,
            "condition": condition,
            "protocol": result['answer'],
            "sources": result['sources'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Protocol retrieval error: {str(e)}")


# Statistics endpoint
@app.get("/api/stats")
async def get_statistics():
    """Get usage statistics"""
    return {
        "active_sessions": len(sessions),
        "total_interactions": sum(s.get("message_count", 0) for s in sessions.values()),
        "supported_languages": len(config.SUPPORTED_LANGUAGES),
        "timestamp": datetime.now().isoformat()
    }


# Run server
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # Changed from "info" to "debug"
    )