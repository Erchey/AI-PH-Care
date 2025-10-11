"""
Healthcare AI Agent - AI-Driven Main Structure with RAG
Built with LangGraph and Groq for Primary Healthcare Support
Uses RAG for dynamic medical knowledge retrieval
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from healthcare_config import Config
from tools import get_all_tools
from modules.rag_system import MedicalRAGSystem
from modules.triage import AITriageModule
from modules.referral import AIReferralModule
from modules.resource_allocation import AIResourceModule
from modules.patient_routing import AIRoutingModule
from langsmith import Client

# Initialize client (optional — for advanced control)
client = Client()

class AgentState(TypedDict):
    """State of the healthcare agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_patient: dict
    context: dict
    language: str
    offline_mode: bool
    retrieved_docs: list
    decision_history: list


class HealthcareAgent:
    """
    AI-Driven Healthcare Agent for PHC Workers
    Uses LLM reasoning + RAG for intelligent decision making
    """
    
    def __init__(self):
        self.config = Config()
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=self.config.GROQ_MODEL,
            api_key=self.config.GROQ_API_KEY,
            temperature=0.5,  # Low temperature for medical accuracy
            max_tokens=4096
        )
        
        # Initialize RAG system
        self.rag_system = MedicalRAGSystem(self.llm)
        
        # Initialize AI modules
        self.triage_module = AITriageModule(self.llm, self.rag_system)
        self.referral_module = AIReferralModule(self.llm, self.rag_system)
        self.resource_module = AIResourceModule(self.llm, self.rag_system)
        self.routing_module = AIRoutingModule(self.llm, self.rag_system)
        
        # Get tools
        self.tools = get_all_tools(self.rag_system)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classifier", self.classifier_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("triage", self.triage_node)
        workflow.add_node("referral", self.referral_node)
        workflow.add_node("resource_allocation", self.resource_node)
        workflow.add_node("patient_routing", self.routing_node)
        workflow.add_node("summarizer", self.summarizer_node)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Classifier routes to appropriate module
        workflow.add_conditional_edges(
            "classifier",
            self.route_request,
            {
                "triage": "triage",
                "referral": "referral",
                "resource": "resource_allocation",
                "routing": "patient_routing",
                "general": "agent",
            }
        )
        
        # Each module can use agent for additional reasoning
        workflow.add_edge("triage", "agent")
        workflow.add_edge("referral", "agent")
        workflow.add_edge("resource_allocation", "agent")
        workflow.add_edge("patient_routing", "agent")
        
        # Agent can use tools or finish
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "summarize": "summarizer",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarizer", END)
        
        return workflow.compile()
    
    def classifier_node(self, state: AgentState):
        """AI classifier to route requests intelligently"""
        
        messages = state["messages"]
        user_message = messages[-1].content if messages else ""
        patient_data = state.get("current_patient", {})
        
        classification_prompt = f"""You are a healthcare request classifier. Analyze the request and determine the appropriate action.

User Request: {user_message}

Patient Context: {patient_data}

Available Actions:
- triage: Patient needs urgent assessment, prioritization, or severity evaluation
- referral: Need to refer patient to specialist or another facility
- resource: Resource allocation, bed management, staff scheduling
- routing: Determine which department/provider should see patient
- general: General medical inquiry, protocol question, or information lookup

Respond with ONLY the action name (triage/referral/resource/routing/general).
"""
        
        response = self.llm.invoke(classification_prompt)
        state["decision_history"] = state.get("decision_history", [])
        state["decision_history"].append({
            "step": "classification",
            "result": response.content.strip().lower()
        })
        
        return state
    
    def route_request(self, state: AgentState):
        """Route based on classification"""
        decision = state.get("decision_history", [{}])[-1].get("result", "general")
        
        # Map classification to valid routes
        route_map = {
            "triage": "triage",
            "referral": "referral",
            "resource": "resource",
            "routing": "routing",
            "general": "general"
        }
        
        return route_map.get(decision, "general")
    
    def agent_node(self, state: AgentState):
        """Main AI agent with reasoning"""
        
        messages = state["messages"]
        language = state.get("language", "English")
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Build context-aware system message
        system_msg = self._build_system_message(state, retrieved_docs)
        
        # Prepare messages with context
        context_messages = [system_msg] + list(messages)
        
        # Get LLM response with tools
        response = self.llm_with_tools.invoke(context_messages)
        
        return {"messages": [response]}
    
    def triage_node(self, state: AgentState):
        """AI-driven triage assessment"""
        
        result = self.triage_module.assess_patient(
            patient_data=state.get("current_patient", {}),
            user_query=state["messages"][-1].content,
            language=state.get("language", "English")
        )
        
        message = AIMessage(content=result["explanation"])
        
        return {
            "messages": [message],
            "context": {**state.get("context", {}), "triage": result},
            "retrieved_docs": result.get("sources", [])
        }
    
    def referral_node(self, state: AgentState):
        """AI-driven referral guidance"""
        
        result = self.referral_module.generate_referral(
            patient_data=state.get("current_patient", {}),
            context=state.get("context", {}),
            language=state.get("language", "English")
        )
        
        message = AIMessage(content=result["recommendation"])
        
        return {
            "messages": [message],
            "context": {**state.get("context", {}), "referral": result},
            "retrieved_docs": result.get("sources", [])
        }
    
    def resource_node(self, state: AgentState):
        """AI-driven resource allocation"""
        
        result = self.resource_module.optimize_resources(
            facility_data=state.get("context", {}).get("facility", {}),
            current_demand=state.get("current_patient", {}),
            language=state.get("language", "English")
        )
        
        message = AIMessage(content=result["recommendations"])
        
        return {"messages": [message], "context": {**state.get("context", {}), "resources": result}}
    
    def routing_node(self, state: AgentState):
        """AI-driven patient routing"""
        
        result = self.routing_module.route_patient(
            patient_data=state.get("current_patient", {}),
            facility_context=state.get("context", {}).get("facility", {}),
            language=state.get("language", "English")
        )
        
        message = AIMessage(content=result["routing_plan"])
        
        return {"messages": [message], "context": {**state.get("context", {}), "routing": result}}
    
    def summarizer_node(self, state: AgentState):
        """Summarize the interaction with action items"""
        
        messages = state["messages"]
        context = state.get("context", {})
        
        summary_prompt = f"""Summarize this healthcare interaction clearly and concisely.

Conversation History:
{self._format_messages(messages)}

Context: {context}

Provide:
1. Brief summary of the case
2. Key decisions made
3. Action items for the healthcare worker
4. Any safety alerts or follow-up needed

Format in clear, actionable language for PHC workers.
"""
        
        summary = self.llm.invoke(summary_prompt)
        
        return {"messages": [AIMessage(content=summary.content)]}
    
    def should_continue(self, state: AgentState):
        """Determine if agent should continue, use tools, or end"""
        
        last_message = state["messages"][-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we need summarization
        if len(state["messages"]) > 8:
            return "summarize"
        
        # Check if response is complete
        content = last_message.content.lower() if hasattr(last_message, 'content') else ""
        
        if any(phrase in content for phrase in ["is there anything else", "any other questions", "help with anything else"]):
            return "end"
        
        return "end"
    
    def _build_system_message(self, state: AgentState, retrieved_docs: list):
        """Build context-aware system message"""
        
        language = state.get("language", "English")
        offline = state.get("offline_mode", False)
        
        context_info = ""
        if retrieved_docs:
            context_info = "\n\nRelevant Medical Knowledge:\n" + "\n".join([
                f"- {doc['content'][:200]}..." for doc in retrieved_docs[:3]
            ])
        
        prompt = f"""You are an AI healthcare assistant supporting Primary Healthcare (PHC) workers in resource-limited settings.

Language: {language}
Mode: {"Offline" if offline else "Online"}

Your role:
- Provide evidence-based medical guidance
- Support clinical decision-making
- Be culturally sensitive and use simple language
- Always prioritize patient safety
- Acknowledge uncertainty and recommend escalation when appropriate

You have access to:
- Medical protocols and guidelines (via RAG)
- Drug databases and formularies
- Emergency procedures
- Facility and resource information

CRITICAL: For serious conditions, always recommend consulting a doctor or senior healthcare professional. You are a decision support tool, not a replacement for clinical judgment.

{context_info}
"""
        
        return SystemMessage(content=prompt)
    
    def _format_messages(self, messages):
        """Format messages for summarization"""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Healthcare Worker: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"AI Assistant: {msg.content}")
        return "\n".join(formatted)
    
    def run(self, user_input: str, **kwargs):
        """
        Run the healthcare agent
        
        Args:
            user_input: Healthcare worker's query
            **kwargs: Additional context (patient_data, facility_data, language, etc.)
        """
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "current_patient": kwargs.get("patient_data", {}),
            "context": {
                "facility": kwargs.get("facility_data", {}),
                **kwargs.get("additional_context", {})
            },
            "language": kwargs.get("language", "English"),
            "offline_mode": kwargs.get("offline_mode", False),
            "retrieved_docs": [],
            "decision_history": []
        }
        
        result = self.graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    agent = HealthcareAgent()
    
    # Example: Dynamic triage
    patient = {
        "age": 4,
        "gender": "female",
        "chief_complaint": "blur vision",
        "vital_signs": {
            "blood_pressure": "160/95",
            "heart_rate": 110,
            "respiratory_rate": 24,
            "oxygen_saturation": 93,
            "temperature": 37.2
        }
    }
    
    result = agent.run(
        user_input="I have a 4 year-old woman partially blind. Help me assess the urgency.",
        patient_data=patient,
        language="igbo"
    )
    
    print("\n=== AI Agent Response ===")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content}\n")
