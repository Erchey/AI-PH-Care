"""
AI-Driven Referral Module
Uses LLM reasoning for intelligent specialist referral recommendations
"""

from typing import Dict
import json


class AIReferralModule:
    """AI-powered referral system"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
    
    def generate_referral(self, patient_data: Dict, context: Dict, language: str = "English") -> Dict:
        """
        Generate AI-driven referral recommendation
        
        Args:
            patient_data: Patient information
            context: Additional context (triage results, facility data)
            language: Preferred language
        
        Returns:
            Comprehensive referral recommendation
        """
        
        # Retrieve referral guidelines
        guidelines = self._retrieve_referral_guidelines(patient_data, context)
        
        # Build referral prompt
        prompt = self._build_referral_prompt(patient_data, context, guidelines, language)
        
        # Get AI recommendation
        response = self.llm.invoke(prompt)
        
        # Structure response
        result = self._structure_referral_response(response.content, patient_data)
        result['sources'] = guidelines
        
        return result
    
    def _retrieve_referral_guidelines(self, patient_data: Dict, context: Dict) -> list:
        """Retrieve relevant referral criteria using RAG"""
        
        chief_complaint = patient_data.get('chief_complaint', '')
        symptoms = ', '.join(patient_data.get('symptoms', []))
        triage_priority = context.get('triage', {}).get('priority', '')
        
        query = f"""referral criteria guidelines for {chief_complaint} {symptoms} 
                    when to refer specialist referral indications urgency {triage_priority}"""
        
        docs = self.rag_system.retrieve(query, k=3)
        return docs
    
    def _build_referral_prompt(
        self,
        patient_data: Dict,
        context: Dict,
        guidelines: list,
        language: str
    ) -> str:
        """Build comprehensive referral assessment prompt"""
        
        age = patient_data.get('age', 'unknown')
        gender = patient_data.get('gender', 'unknown')
        chief_complaint = patient_data.get('chief_complaint', '')
        symptoms = patient_data.get('symptoms', [])
        medical_history = patient_data.get('medical_history', [])
        vital_signs = patient_data.get('vital_signs', {})
        
        triage_info = context.get('triage', {})
        facility_info = context.get('facility', {})
        
        guidelines_text = "\n\n".join([
            f"Guideline {i+1}:\n{doc['content']}"
            for i, doc in enumerate(guidelines)
        ]) if guidelines else "No specific guidelines retrieved"
        
        prompt = f"""You are an expert clinical advisor helping a PHC worker determine if a patient needs specialist referral.

PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- Chief Complaint: {chief_complaint}
- Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
- Medical History: {', '.join(medical_history) if medical_history else 'None'}
- Vital Signs: {json.dumps(vital_signs, indent=2)}

TRIAGE ASSESSMENT:
- Priority: {triage_info.get('priority', 'Not assessed')}
- Key Concerns: {triage_info.get('explanation', '')[:200] if triage_info else 'N/A'}

CURRENT FACILITY CAPABILITIES:
{json.dumps(facility_info, indent=2) if facility_info else 'Limited PHC facility'}

CLINICAL GUIDELINES:
{guidelines_text}

YOUR TASK:
Assess whether this patient needs referral and provide detailed recommendations.

1. **REFERRAL DECISION** (Choose ONE):
   - IMMEDIATE TRANSFER: Life-threatening, transfer now via ambulance
   - URGENT REFERRAL: Needs specialist within 24 hours
   - ROUTINE REFERRAL: Schedule specialist appointment within 1-2 weeks
   - MANAGE AT PHC: Can be managed at primary care level
   - REFER IF NO IMPROVEMENT: Try PHC management, refer if not improving

2. **REASONING** (3-4 sentences):
   - Why referral is/isn't needed
   - What specific concern requires specialist care
   - Why PHC cannot manage (if applicable)

3. **RECOMMENDED SPECIALTY** (if referral needed):
   - Which specialist (cardiology, surgery, etc.)
   - Why this specialty specifically

4. **PRE-REFERRAL MANAGEMENT**:
   - What to do BEFORE transfer/while waiting
   - Medications to start
   - Stabilization measures
   - Tests to order if available

5. **REFERRAL URGENCY INDICATORS**:
   - Signs that mean IMMEDIATE transfer needed
   - Red flags requiring escalation

6. **INFORMATION FOR SPECIALIST**:
   - Key clinical findings to communicate
   - Important history points
   - Results of any tests done

7. **TRANSPORT RECOMMENDATION**:
   - How should patient travel (ambulance, private vehicle, walk)
   - Who should accompany patient
   - Special precautions during transport

8. **ALTERNATIVE IF REFERRAL NOT POSSIBLE**:
   - What if no specialist available nearby
   - PHC management strategies
   - Telemedicine consultation options

IMPORTANT CONSIDERATIONS:
- This is a resource-limited setting
- Balance ideal care with practical reality
- Patient may face transport challenges
- Consider cost implications for patient
- Provide culturally appropriate guidance
- Use language: {language}

Provide practical, actionable recommendations.
"""
        
        return prompt
    
    def _structure_referral_response(self, ai_response: str, patient_data: Dict) -> Dict:
        """Structure AI referral response"""
        
        decision = self._extract_referral_decision(ai_response)
        
        result = {
            "referral_decision": decision,
            "recommendation": ai_response,
            "requires_immediate_transfer": decision == "IMMEDIATE_TRANSFER",
            "urgency_level": self._map_urgency(decision),
            "patient_summary": {
                "age": patient_data.get('age'),
                "chief_complaint": patient_data.get('chief_complaint', ''),
            }
        }
        
        return result
    
    def _extract_referral_decision(self, response: str) -> str:
        """Extract referral decision from response"""
        
        response_upper = response.upper()
        
        if "IMMEDIATE TRANSFER" in response_upper or "IMMEDIATE REFERRAL" in response_upper:
            return "IMMEDIATE_TRANSFER"
        elif "URGENT REFERRAL" in response_upper:
            return "URGENT_REFERRAL"
        elif "ROUTINE REFERRAL" in response_upper:
            return "ROUTINE_REFERRAL"
        elif "REFER IF NO IMPROVEMENT" in response_upper:
            return "CONDITIONAL_REFERRAL"
        elif "MANAGE AT PHC" in response_upper or "NO REFERRAL" in response_upper:
            return "MANAGE_AT_PHC"
        else:
            return "NEEDS_ASSESSMENT"
    
    def _map_urgency(self, decision: str) -> str:
        """Map decision to urgency level"""
        
        urgency_map = {
            "IMMEDIATE_TRANSFER": "CRITICAL",
            "URGENT_REFERRAL": "URGENT",
            "ROUTINE_REFERRAL": "ROUTINE",
            "CONDITIONAL_REFERRAL": "CONDITIONAL",
            "MANAGE_AT_PHC": "NOT_NEEDED",
            "NEEDS_ASSESSMENT": "UNCERTAIN"
        }
        
        return urgency_map.get(decision, "UNCERTAIN")
    
    def generate_referral_letter(
        self,
        patient_data: Dict,
        referral_recommendation: Dict,
        language: str = "English"
    ) -> str:
        """
        Generate formal referral letter for specialist
        
        Args:
            patient_data: Patient information
            referral_recommendation: Referral assessment results
            language: Language for letter
        
        Returns:
            Formatted referral letter
        """
        
        letter_prompt = f"""Generate a professional medical referral letter.

PATIENT: {json.dumps(patient_data, indent=2)}

REFERRAL RECOMMENDATION: {referral_recommendation.get('recommendation', '')[:500]}

Create a formal referral letter including:
1. Patient demographics
2. Reason for referral (presenting complaint)
3. Relevant history
4. Clinical findings
5. Investigations performed
6. Current treatment
7. Specific question for specialist
8. Urgency level

Format as a professional medical letter.
Language: {language}
"""
        
        response = self.llm.invoke(letter_prompt)
        return response.content
    
    def assess_referral_appropriateness(
        self,
        patient_data: Dict,
        proposed_specialty: str,
        language: str = "English"
    ) -> str:
        """
        Assess if a proposed referral is appropriate
        Useful for training or second opinion
        
        Args:
            patient_data: Patient information
            proposed_specialty: Specialty healthcare worker wants to refer to
            language: Language preference
        
        Returns:
            Assessment of referral appropriateness
        """
        
        assessment_prompt = f"""A PHC worker wants to refer this patient to {proposed_specialty}.
Assess if this referral is appropriate.

PATIENT: {json.dumps(patient_data, indent=2)}

PROPOSED REFERRAL TO: {proposed_specialty}

Provide:
1. Is this referral appropriate? (Yes/No/Uncertain)
2. Reasoning
3. If inappropriate, suggest correct referral or management
4. If appropriate, confirm it's the right specialty
5. Any additional workup needed before referral

Language: {language}
"""
        
        response = self.llm.invoke(assessment_prompt)
        return response.content
