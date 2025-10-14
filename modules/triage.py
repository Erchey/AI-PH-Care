"""
AI-Driven Triage Module
Uses LLM reasoning and RAG for intelligent patient triage
No hardcoded rules - fully AI-powered decision making
"""

from typing import Dict, Optional
import json


class AITriageModule:
    """AI-powered triage system using LLM reasoning"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
    
    def assess_patient(self, patient_data: Optional[Dict], user_query: str, language: str = "English") -> Dict:
        """
        Perform AI-driven triage assessment
        
        Args:
            patient_data: Patient information (can be None)
            user_query: Healthcare worker's query
            language: Preferred language
        
        Returns:
            Comprehensive triage assessment with reasoning
        """
        
        # Handle None patient_data
        if patient_data is None:
            patient_data = {}
        
        # Validate patient_data is a dictionary
        if not isinstance(patient_data, dict):
            patient_data = {}
        
        # Retrieve relevant triage guidelines using RAG
        triage_guidelines = self._retrieve_triage_guidelines(patient_data)
        
        # Build comprehensive triage prompt
        triage_prompt = self._build_triage_prompt(
            patient_data,
            user_query,
            triage_guidelines,
            language
        )
        
        # Get AI assessment
        response = self.llm.invoke(triage_prompt)
        
        # Parse and structure the response
        structured_result = self._structure_response(response.content, patient_data)
        
        # Add metadata
        structured_result['sources'] = triage_guidelines
        structured_result['language'] = language
        
        return structured_result
    
    def _retrieve_triage_guidelines(self, patient_data: Dict) -> list:
        """Retrieve relevant triage guidelines using RAG"""
        
        # Safely extract patient information with defaults
        symptoms = patient_data.get('symptoms', []) if patient_data else []
        chief_complaint = patient_data.get('chief_complaint', '') if patient_data else ''
        age = patient_data.get('age', 'adult') if patient_data else 'adult'
        
        # Ensure symptoms is a list
        if not isinstance(symptoms, list):
            symptoms = []
        
        # Build fallback query if no specific data
        if not chief_complaint and not symptoms:
            query = "general triage assessment guidelines priority determination urgency"
        else:
            symptoms_str = ', '.join(str(s) for s in symptoms) if symptoms else 'general symptoms'
            query = f"""triage assessment guidelines for patient with {chief_complaint or 'complaint'} 
                        symptoms {symptoms_str} age {age} priority determination urgency assessment"""
        
        try:
            # Retrieve relevant documents
            docs = self.rag_system.retrieve(query, k=3)
            return docs if docs else []
        except Exception as e:
            print(f"Warning: Could not retrieve triage guidelines: {e}")
            return []
    
    def _build_triage_prompt(
        self,
        patient_data: Dict,
        user_query: str,
        guidelines: list,
        language: str
    ) -> str:
        """Build comprehensive prompt for AI triage"""
        
        # Safely extract patient information with defaults
        age = patient_data.get('age', 'unknown') if patient_data else 'unknown'
        gender = patient_data.get('gender', 'unknown') if patient_data else 'unknown'
        chief_complaint = patient_data.get('chief_complaint', user_query) if patient_data else user_query
        symptoms = patient_data.get('symptoms', []) if patient_data else []
        vital_signs = patient_data.get('vital_signs', {}) if patient_data else {}
        medical_history = patient_data.get('medical_history', []) if patient_data else []
        
        # Ensure proper types
        if not isinstance(symptoms, list):
            symptoms = []
        if not isinstance(vital_signs, dict):
            vital_signs = {}
        if not isinstance(medical_history, list):
            medical_history = []
        
        # Format vital signs
        vital_signs_str = json.dumps(vital_signs, indent=2) if vital_signs else "Not available"
        
        # Format guidelines
        guidelines_text = "\n\n".join([
            f"Guideline {i+1}:\n{doc.get('content', 'No content')}"
            for i, doc in enumerate(guidelines)
        ]) if guidelines else "No specific guidelines retrieved"
        
        prompt = f"""You are an expert healthcare AI performing patient triage assessment for a Primary Healthcare Center.

PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- Chief Complaint: {chief_complaint}
- Symptoms: {', '.join(str(s) for s in symptoms) if symptoms else 'Not specified'}
- Medical History: {', '.join(str(h) for h in medical_history) if medical_history else 'None reported'}

VITAL SIGNS:
{vital_signs_str}

HEALTHCARE WORKER QUERY:
{user_query}

RELEVANT CLINICAL GUIDELINES:
{guidelines_text}

YOUR TASK:
Perform a comprehensive triage assessment and provide:

1. **Priority Level** (choose ONE):
   - CRITICAL (Life-threatening, immediate intervention needed)
   - URGENT (Serious condition, needs prompt attention within 15 min)
   - SEMI-URGENT (Moderate condition, can wait 30-60 min)
   - NON-URGENT (Stable, routine care appropriate)

2. **Clinical Reasoning** (2-3 sentences):
   - Explain why you assigned this priority
   - Highlight the most concerning findings
   - Note any red flags or protective factors

3. **Immediate Actions** (3-5 specific steps):
   - What should the healthcare worker do RIGHT NOW
   - Any urgent interventions needed
   - Monitoring requirements

4. **Red Flags to Monitor**:
   - Warning signs that would require immediate escalation
   - Symptoms indicating deterioration

5. **Recommended Timeline**:
   - When should a doctor see this patient
   - Acceptable wait time

6. **Risk Assessment**:
   - Short-term risks (next few hours)
   - Factors that increase urgency

IMPORTANT GUIDELINES:
- Base your assessment on clinical evidence and the guidelines provided
- When in doubt, err on the side of caution - prioritize patient safety
- Consider age-specific risks (pediatric, elderly)
- Account for resource-limited settings
- Provide actionable, practical recommendations
- Use language appropriate for PHC workers in {language}
- If patient information is limited, acknowledge this and provide conservative recommendations

Provide your assessment in a clear, structured format that a PHC worker can follow immediately.
"""
        
        return prompt
    
    def _structure_response(self, ai_response: str, patient_data: Optional[Dict]) -> Dict:
        """Structure the AI response into standardized format"""
        
        # Ensure patient_data is a dict
        if not patient_data:
            patient_data = {}
        
        # Extract priority level from response
        priority = self._extract_priority(ai_response)
        
        # Structure the result
        result = {
            "priority": priority,
            "explanation": ai_response,
            "patient_summary": {
                "age": patient_data.get('age', 'unknown'),
                "chief_complaint": patient_data.get('chief_complaint', ''),
                "key_vitals": patient_data.get('vital_signs', {})
            },
            "requires_immediate_doctor": priority in ["CRITICAL", "URGENT"],
            "estimated_wait_minutes": self._get_wait_time(priority),
            "safety_net_advice": self._generate_safety_net(priority)
        }
        
        return result
    
    def _extract_priority(self, response: str) -> str:
        """Extract priority level from AI response"""
        
        if not response:
            return "URGENT"  # Default to URGENT if no response
        
        response_upper = response.upper()
        
        # Check for priority keywords in order of severity
        if "CRITICAL" in response_upper:
            return "CRITICAL"
        elif "URGENT" in response_upper:
            return "URGENT"
        elif "SEMI-URGENT" in response_upper or "SEMI URGENT" in response_upper:
            return "SEMI_URGENT"
        elif "NON-URGENT" in response_upper or "NON URGENT" in response_upper:
            return "NON_URGENT"
        else:
            # Default to URGENT if unclear (safety first)
            return "URGENT"
    
    def _get_wait_time(self, priority: str) -> int:
        """Get appropriate wait time for priority level"""
        
        wait_times = {
            "CRITICAL": 0,
            "URGENT": 15,
            "SEMI_URGENT": 60,
            "NON_URGENT": 120
        }
        
        return wait_times.get(priority, 60)
    
    def _generate_safety_net(self, priority: str) -> str:
        """Generate safety net advice based on priority"""
        
        safety_nets = {
            "CRITICAL": "If patient's condition worsens AT ALL, escalate immediately. Consider transfer to higher-level facility.",
            "URGENT": "Monitor closely. If symptoms worsen or new symptoms develop, reassess immediately and escalate.",
            "SEMI_URGENT": "Advise patient to return immediately if symptoms worsen, new symptoms develop, or they feel significantly worse.",
            "NON_URGENT": "Provide patient education on warning signs. Advise to return if condition doesn't improve in 24-48 hours or worsens."
        }
        
        return safety_nets.get(priority, "Monitor patient and reassess if condition changes.")
    
    def compare_triage_decisions(
        self, 
        patient_data: Optional[Dict], 
        human_triage: str, 
        language: str = "English"
    ) -> str:
        """
        Compare AI triage with human clinician's triage for learning
        
        Args:
            patient_data: Patient information (can be None)
            human_triage: Human clinician's triage decision
            language: Language preference
        
        Returns:
            Analysis comparing the two decisions
        """
        
        # Ensure patient_data is not None
        if patient_data is None:
            patient_data = {}
        
        # Get AI assessment
        ai_assessment = self.assess_patient(patient_data, "Assess triage priority", language)
        
        comparison_prompt = f"""Compare these two triage assessments and provide educational feedback.

PATIENT: {json.dumps(patient_data, indent=2)}

HUMAN CLINICIAN TRIAGE: {human_triage}

AI TRIAGE: {ai_assessment['priority']}
Reasoning: {ai_assessment['explanation'][:500]}

Provide:
1. Agreement or disagreement between assessments
2. If different, explain which factors led to different conclusions
3. Educational points for the clinician
4. Any additional considerations

Keep feedback constructive and educational.
"""
        
        response = self.llm.invoke(comparison_prompt)
        return response.content
    
    def get_triage_training(self, scenario: str, language: str = "English") -> str:
        """
        Generate triage training scenario and explanation
        
        Args:
            scenario: Clinical scenario description
            language: Language preference
        
        Returns:
            Educational triage walkthrough
        """
        
        training_prompt = f"""You are a triage educator. Walk through this clinical scenario step-by-step.

SCENARIO: {scenario}

Provide:
1. How to approach this triage assessment
2. Key clinical findings to look for
3. What priority level would be appropriate and why
4. Common pitfalls to avoid
5. Learning points

Make it educational and practical for PHC workers.
Language: {language}
"""
        
        response = self.llm.invoke(training_prompt)
        return response.content