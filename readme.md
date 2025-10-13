# 🏥 Healthcare AI Agent for PHC Workers

An intelligent AI agent built with **LangGraph** and **Groq** to support Primary Healthcare (PHC) workers in resource-limited settings. Uses RAG (Retrieval-Augmented Generation) for evidence-based medical guidance.

## 🎯 Hackathon Objectives

✅ **Intelligent, Accessible PHC Support Tools**
- AI-assisted triage and patient assessment
- Natural language interfaces
- Multilingual support (8+ languages)
- Offline-capable design

✅ **Integrated PHC Decision-Support**
- Real-time patient routing optimization
- Intelligent referral guidance
- Resource allocation recommendations
- Evidence-based clinical protocols via RAG

## 🌟 Key Features

### 1. **AI-Driven Triage** 🚨
- Intelligent patient priority assessment
- Real-time vital signs interpretation
- Red flag identification
- Safety-first reasoning with LLM

### 2. **Smart Referral System** 🏨
- Automated referral recommendations
- Specialty matching
- Pre-referral stabilization guidance
- Referral letter generation

### 3. **Resource Optimization** 📊
- Bed allocation intelligence
- Staff deployment optimization
- Supply chain alerts
- Ethical resource triage

### 4. **Patient Routing** 🗺️
- Optimal facility flow paths
- Queue management
- Provider matching
- Transfer assessment

### 5. **RAG Medical Knowledge** 📚
- Evidence-based guidance from documents
- Protocol and guideline retrieval
- Drug information lookup
- Diagnostic support

### 6. **Multilingual Support** 🌍
- English, Spanish, French, Portuguese
- Swahili, Hindi, Arabic, Mandarin
- Context-aware translations

## 🏗️ Architecture

```
healthcare-ai-agent/
│
├── agent.py                    # Main LangGraph agent
├── config.py                   # Configuration management
├── tools.py                    # AI-powered tools with RAG
│
├── modules/
│   ├── rag_system.py          # Medical knowledge RAG
│   ├── triage.py              # AI triage module
│   ├── referral.py            # AI referral module
│   ├── resource_allocation.py # Resource optimization
│   └── patient_routing.py     # Patient flow optimization
│
├── medical_docs/              # Medical documents for RAG
│   ├── protocols/
│   ├── guidelines/
│   └── formularies/
│
├── vector_store/              # Chroma vector database
├── demo.py                    # Usage examples
├── requirements.txt           # Dependencies
└── .env                       # Configuration (create from .env.example)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Erchey/AI-PH-Care.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env

# Edit .env and add your Groq API key
# Get free API key at: https://console.groq.com
```

### 3. Setup Medical Knowledge Base

```python
from modules.rag_system import MedicalRAGSystem, setup_initial_knowledge_base
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-70b-versatile", api_key="your-key")
rag = MedicalRAGSystem(llm)

# Ingest your medical documents
rag.ingest_document("./medical_docs/who_guidelines.pdf", 
                    metadata={"category": "guidelines", "org": "WHO"})

# Or ingest from URLs
rag.ingest_document("https://www.who.int/publications/guidelines",
                    metadata={"source": "WHO", "type": "guidelines"})
```

### 4. Run Demo

```bash
python demo.py
```

## 💡 Usage Examples

### Example 1: Triage Assessment

```python
from agent import HealthcareAgent

agent = HealthcareAgent()

patient = {
    "age": 45,
    "gender": "female",
    "chief_complaint": "severe chest pain and shortness of breath",
    "vital_signs": {
        "blood_pressure": "160/95",
        "heart_rate": 110,
        "oxygen_saturation": 93
    },
    "medical_history": ["hypertension", "diabetes"]
}

result = agent.run(
    user_input="Assess this patient's urgency",
    patient_data=patient,
    language="English"
)

# AI provides priority level, reasoning, and immediate actions
```

### Example 2: Referral Guidance

```python
result = agent.run(
    user_input="Should I refer this patient to cardiology?",
    patient_data=patient,
    facility_data={"specialists": [], "capabilities": ["basic lab"]},
    language="English"
)

# AI analyzes and recommends referral with reasoning
```

### Example 3: Resource Allocation

```python
facility_data = {
    "beds": {"total": 10, "occupied": 9},
    "staff": {"on_duty": 3},
    "patients": {"waiting": 15}
}

result = agent.run(
    user_input="Help optimize our limited resources",
    facility_data=facility_data
)

# AI provides resource allocation strategy
```

### Example 4: RAG Medical Query

```python
from modules.rag_system import MedicalRAGSystem

rag = MedicalRAGSystem(llm)

# Query medical knowledge base
answer = rag.answer_with_sources(
    "What is the treatment protocol for severe malaria?"
)

print(answer['answer'])
print("Sources:", answer['sources'])
```

## 🛠️ Customization

### Adding New Medical Documents

```python
# PDF documents
rag.ingest_document("path/to/medical_protocol.pdf", 
                    metadata={"category": "emergency"})

# Web resources
rag.ingest_document("https://medical-guidelines-url.com",
                    metadata={"source": "Ministry of Health"})

# Markdown files
rag.ingest_document("protocols/malaria_treatment.md",
                    metadata={"disease": "malaria"})
```


### Multilingual Configuration

```python
# Agent automatically handles multiple languages
result = agent.run(
    user_input="Paciente con fiebre alta",
    patient_data=patient,
    language="Spanish"  # or "French", "Swahili", etc.
)
```

## 🎓 Module Details

### AI Triage Module
- **Zero hardcoded rules** - Pure LLM reasoning
- RAG retrieval of triage protocols
- Context-aware priority assignment
- Safety-first approach

### AI Referral Module
- Intelligent specialty matching
- Evidence-based referral criteria
- Pre-referral care guidance
- Automated referral letters

### Resource Allocation Module
- Ethical resource distribution
- Capacity planning
- Staff optimization
- Supply chain management

### Patient Routing Module
- Optimal patient flow paths
- Queue optimization
- Provider matching
- Transfer assessment

### RAG System
- Vector-based document retrieval
- Multi-format support (PDF, Web, Markdown)
- Source attribution
- Semantic search

## 📊 API Endpoints (Optional)

For web deployment, wrap the agent:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/triage")
async def triage_patient(patient_data: dict):
    agent = HealthcareAgent()
    result = agent.run(
        user_input="Perform triage assessment",
        patient_data=patient_data
    )
    return result
```

## 🔒 Security & Privacy

- Patient data stays local (no external transmission)
- HIPAA-compliant design considerations
- Configurable data retention
- Audit logging capability

## 🌐 Offline Capability

```python
# Configure offline mode
result = agent.run(
    user_input="Assess patient",
    patient_data=patient,
    offline_mode=True  # Uses cached embeddings and local models
)
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test specific module
pytest tests/test_triage.py
```

## 📈 Performance

- **Latency**: ~2-5 seconds per query (Groq)
- **Throughput**: Handles concurrent requests
- **Accuracy**: Depends on RAG document quality
- **Languages**: 8+ supported

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add medical documents to RAG
4. Test thoroughly
5. Submit pull request

## 📜 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Built with LangGraph by LangChain
- Powered by Groq's lightning-fast inference
- Medical knowledge from WHO, CDC, and national health authorities


## 🚀 Deployment Options

### Local Deployment
```bash
python agent.py
```

---

**Built for healthcare workers, powered by AI** 🏥🤖
