"""
AI-Driven Patient Routing Module
Intelligently routes patients to appropriate departments/providers
"""

from typing import Dict, List
import json


class AIRoutingModule:
    """AI-powered patient routing system"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
    
    def route_patient(
        self,
        patient_data: Dict,
        facility_context: Dict,
        language: str = "English"
    ) -> Dict:
        """
        Determine optimal patient routing through facility
        
        Args:
            patient_data: Patient information
            facility_context: Facility departments and current status
            language: Preferred language
        
        Returns:
            Routing recommendation with reasoning
        """
        
        # Retrieve routing protocols
        protocols = self._retrieve_routing_protocols(patient_data)
        
        # Build routing prompt
        prompt = self._build_routing_prompt(
            patient_data,
            facility_context,
            protocols,
            language
        )
        
        # Get AI recommendation
        response = self.llm.invoke(prompt)
        
        # Structure response
        result = {
            "routing_plan": response.content,
            "sources": protocols,
            "recommended_provider": self._extract_provider_type(response.content),
            "estimated_duration": self._estimate_visit_duration(patient_data),
            "special_considerations": self._identify_special_needs(patient_data)
        }
        
        return result
    
    def _retrieve_routing_protocols(self, patient_data: Dict) -> list:
        """Retrieve patient flow protocols using RAG"""
        
        chief_complaint = patient_data.get('chief_complaint', '')
        symptoms = ', '.join(patient_data.get('symptoms', []))
        
        query = f"""patient flow protocols department routing for {chief_complaint} {symptoms} 
                   which provider should see patient triage destination"""
        
        docs = self.rag_system.retrieve(query, k=2)
        return docs
    
    def _build_routing_prompt(
        self,
        patient_data: Dict,
        facility_context: Dict,
        protocols: list,
        language: str
    ) -> str:
        """Build patient routing prompt"""
        
        protocols_text = "\n\n".join([
            f"Protocol {i+1}:\n{doc['content']}"
            for i, doc in enumerate(protocols)
        ]) if protocols else "Using standard triage protocols"
        
        prompt = f"""You are a patient flow coordinator determining the best routing for a patient in a PHC facility.

PATIENT INFORMATION:
{json.dumps(patient_data, indent=2)}

FACILITY STRUCTURE & STATUS:
{json.dumps(facility_context, indent=2)}

ROUTING PROTOCOLS:
{protocols_text}

YOUR TASK:
Determine the optimal path for this patient through the healthcare facility.

Provide:

1. **INITIAL DESTINATION**:
   - Where should patient go first?
   - Triage area, direct to nurse, urgent care, etc.
   - Reasoning for this choice

2. **PROVIDER TYPE NEEDED**:
   - Nurse assessment only
   - Clinical Officer
   - Medical Doctor
   - Specialist (if available)
   - Why this level of provider?

3. **QUEUE/PRIORITY**:
   - Which queue (emergency, urgent, routine)
   - Estimated wait time
   - Can they wait safely?

4. **ASSESSMENT STATIONS**:
   - Vitals station
   - Lab/testing needed
   - Imaging needed
   - Order of stations

5. **WORKFLOW OPTIMIZATION**:
   - Parallel processing opportunities
   - What can be done while waiting
   - Efficient sequencing

6. **SPECIAL ACCOMMODATIONS**:
   - Mobility assistance needed
   - Isolation requirements
   - Language interpreter
   - Pediatric/geriatric considerations

7. **ESCALATION PATH**:
   - If condition worsens while waiting
   - Fast-track criteria
   - When to bypass normal flow

8. **EXPECTED JOURNEY**:
   - Step-by-step patient path
   - Total expected time in facility
   - Discharge or admission likely?

FACILITY CONSTRAINTS:
- Limited staff and rooms
- Patients already waiting
- Resource availability
- Need to optimize flow for everyone

Language: {language}

Provide clear, actionable routing instructions.
"""
        
        return prompt
    
    def _extract_provider_type(self, response: str) -> str:
        """Extract recommended provider type from response"""
        
        response_lower = response.lower()
        
        provider_keywords = {
            "doctor": ["doctor", "physician", "medical doctor", "md"],
            "clinical_officer": ["clinical officer", "clinician"],
            "nurse": ["nurse", "nursing"],
            "specialist": ["specialist", "specialist consultation"],
            "emergency": ["emergency", "resuscitation"]
        }
        
        for provider, keywords in provider_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return provider
        
        return "nurse"  # Default to nurse assessment
    
    def _estimate_visit_duration(self, patient_data: Dict) -> str:
        """Estimate total visit duration"""
        
        # Simple heuristic - can be made more sophisticated
        complexity_indicators = len(patient_data.get('symptoms', [])) + len(patient_data.get('medical_history', []))
        
        if complexity_indicators > 5:
            return "90-120 minutes (complex case)"
        elif complexity_indicators > 3:
            return "45-90 minutes (moderate)"
        else:
            return "20-45 minutes (straightforward)"
    
    def _identify_special_needs(self, patient_data: Dict) -> List[str]:
        """Identify special routing considerations"""
        
        special_needs = []
        
        age = patient_data.get('age', 0)
        if age < 2:
            special_needs.append("Pediatric area - needs parent/guardian")
        elif age > 65:
            special_needs.append("Elderly - may need mobility assistance")
        
        if patient_data.get('gender') == 'female' and patient_data.get('pregnant'):
            special_needs.append("Pregnant - antenatal area if available")
        
        symptoms = [s.lower() for s in patient_data.get('symptoms', [])]
        if any(word in ' '.join(symptoms) for word in ['infectious', 'fever', 'cough', 'rash']):
            special_needs.append("Possible infectious - consider isolation")
        
        if patient_data.get('language') and patient_data.get('language') != 'English':
            special_needs.append(f"Language barrier - {patient_data.get('language')} interpreter needed")
        
        return special_needs
    
    def optimize_queue_management(
        self,
        waiting_patients: List[Dict],
        available_providers: Dict,
        language: str = "English"
    ) -> str:
        """
        Optimize queue and patient assignment to providers
        
        Args:
            waiting_patients: List of patients in queue
            available_providers: Available providers and their specialties
            language: Language preference
        
        Returns:
            Queue optimization recommendations
        """
        
        queue_prompt = f"""Optimize patient queue and provider assignments.

PATIENTS WAITING:
{json.dumps(waiting_patients, indent=2)}

AVAILABLE PROVIDERS:
{json.dumps(available_providers, indent=2)}

Provide:

1. **OPTIMAL QUEUE ORDER**:
   - Recommended seeing order (accounting for urgency and provider match)
   - Reasoning for order

2. **PROVIDER ASSIGNMENTS**:
   - Which provider should see which patient
   - Why they're the best match

3. **PARALLEL PROCESSING**:
   - Patients who can be seen simultaneously
   - Efficiency opportunities

4. **WAIT TIME ESTIMATES**:
   - Realistic wait times for each patient
   - Communication points

5. **QUEUE INTERVENTIONS**:
   - Patients who need reassessment while waiting
   - Those safe to wait vs. need immediate attention

Language: {language}
"""
        
        response = self.llm.invoke(queue_prompt)
        return response.content
    
    def assess_transfer_need(
        self,
        patient_data: Dict,
        current_facility_capabilities: Dict,
        language: str = "English"
    ) -> str:
        """
        Assess if patient needs transfer to higher-level facility
        
        Args:
            patient_data: Patient information
            current_facility_capabilities: What current facility can handle
            language: Language preference
        
        Returns:
            Transfer assessment and recommendations
        """
        
        transfer_prompt = f"""Assess if this patient needs transfer to a higher-level facility.

PATIENT:
{json.dumps(patient_data, indent=2)}

CURRENT FACILITY CAPABILITIES:
{json.dumps(current_facility_capabilities, indent=2)}

Evaluate:

1. **TRANSFER DECISION**:
   - Should patient be transferred? (Yes/No/Uncertain)
   - Level of urgency (Immediate/Urgent/Routine)

2. **REASONING**:
   - Why transfer is/isn't needed
   - What specific care is unavailable here
   - Risks of staying vs. transferring

3. **DESTINATION FACILITY**:
   - Type of facility needed (District Hospital/Regional/Tertiary)
   - Required capabilities

4. **PRE-TRANSFER STABILIZATION**:
   - What must be done before transfer
   - Medications/interventions
   - Monitoring during transport

5. **TRANSPORT METHOD**:
   - Ambulance (basic/advanced)
   - Private vehicle acceptable?
   - Accompaniment needed

6. **COMMUNICATION**:
   - Information for receiving facility
   - Handover details

7. **IF TRANSFER IMPOSSIBLE**:
   - Best management at current facility
   - Telemedicine consultation options
   - Risk mitigation

Language: {language}
"""
        
        response = self.llm.invoke(transfer_prompt)
        return response.content
    
    def generate_patient_flow_analytics(
        self,
        flow_data: Dict,
        language: str = "English"
    ) -> str:
        """
        Analyze patient flow patterns and suggest improvements
        
        Args:
            flow_data: Historical patient flow data
            language: Language preference
        
        Returns:
            Flow analysis and improvement recommendations
        """
        
        analytics_prompt = f"""Analyze patient flow patterns and suggest improvements.

PATIENT FLOW DATA:
{json.dumps(flow_data, indent=2)}

Provide:

1. **BOTTLENECK ANALYSIS**:
   - Where are the delays?
   - What's causing congestion?

2. **EFFICIENCY METRICS**:
   - Average time per stage
   - Overall throughput
   - Wait time analysis

3. **IMPROVEMENT RECOMMENDATIONS**:
   - Quick wins (immediate implementation)
   - Medium-term improvements
   - Systemic changes needed

4. **RESOURCE OPTIMIZATION**:
   - Staff deployment suggestions
   - Space utilization improvements
   - Equipment placement

5. **PATIENT EXPERIENCE**:
   - Pain points in current flow
   - Ways to improve experience

Language: {language}
"""
        
        response = self.llm.invoke(analytics_prompt)
        return response.content
