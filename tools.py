"""
AI-Driven Tools Module with RAG
All tools use LLM reasoning and document retrieval instead of hardcoded data
"""

from langchain.tools import tool
from typing import Dict, List
from langchain_core.tools import StructuredTool


def get_all_tools(rag_system):
    """
    Create tools that use RAG system for dynamic information retrieval
    
    Args:
        rag_system: MedicalRAGSystem instance for document retrieval
    """
    
    @tool
    def search_medical_knowledge(query: str, category: str = None) -> str:
        """
        Search medical knowledge base for protocols, guidelines, or information.
        Use this for any medical question, protocol lookup, or clinical guidance.
        
        Args:
            query: Medical query or topic to search
            category: Optional filter (emergency, drugs, diagnosis, procedures, guidelines)
        
        Returns:
            Relevant medical information with sources
        """
        
        result = rag_system.answer_with_sources(query, k=4)
        
        # Format response with sources
        response = f"{result['answer']}\n\n📚 Sources:\n"
        
        for i, source in enumerate(result['sources'][:3], 1):
            source_name = source['source'].split('/')[-1] if '/' in source['source'] else source['source']
            response += f"{i}. {source_name}\n"
        
        return response
    
    
    @tool
    def get_emergency_protocol(condition: str) -> str:
        """
        Retrieve emergency treatment protocol for a specific condition.
        Use this for life-threatening emergencies requiring immediate action.
        
        Args:
            condition: Emergency condition (e.g., "cardiac arrest", "stroke", "anaphylaxis")
        
        Returns:
            Step-by-step emergency protocol
        """
        
        query = f"emergency protocol immediate treatment for {condition} step by step"
        result = rag_system.answer_with_sources(query, k=3)
        
        return f"🚨 EMERGENCY PROTOCOL: {condition.upper()}\n\n{result['answer']}"
    
    
    @tool
    def lookup_drug_information(drug_name: str, specific_info: str = None) -> str:
        """
        Look up medication information from drug database.
        Use this to check dosing, contraindications, interactions, or side effects.
        
        Args:
            drug_name: Name of the medication
            specific_info: Specific aspect to look up (dosing, contraindications, interactions, etc.)
        
        Returns:
            Comprehensive drug information
        """
        
        if specific_info:
            query = f"{drug_name} {specific_info}"
        else:
            query = f"{drug_name} dosage contraindications side effects drug interactions"
        
        result = rag_system.answer_with_sources(query, k=3)
        
        return f"💊 {drug_name.upper()}\n\n{result['answer']}"
    
    
    @tool
    def get_diagnostic_guidance(symptoms: str, patient_age: int = None) -> str:
        """
        Get diagnostic guidance and differential diagnosis for symptoms.
        Use this to help identify possible conditions based on presentation.
        
        Args:
            symptoms: Patient symptoms description
            patient_age: Patient age (helps narrow differential)
        
        Returns:
            Diagnostic considerations and recommended investigations
        """
        
        age_context = f" in {patient_age} year old patient" if patient_age else ""
        query = f"differential diagnosis and diagnostic approach for {symptoms}{age_context}"
        
        result = rag_system.answer_with_sources(query, k=4)
        
        return f"🔍 DIAGNOSTIC GUIDANCE\n\nSymptoms: {symptoms}\n\n{result['answer']}"
    
    
    @tool
    def search_clinical_guidelines(topic: str, organization: str = None) -> str:
        """
        Search for clinical practice guidelines on a specific topic.
        Use this for evidence-based treatment recommendations.
        
        Args:
            topic: Clinical topic or condition
            organization: Specific guideline organization (WHO, CDC, etc.)
        
        Returns:
            Clinical guidelines and recommendations
        """
        
        org_context = f" from {organization}" if organization else ""
        query = f"clinical practice guidelines for {topic}{org_context} recommendations"
        
        result = rag_system.answer_with_sources(query, k=4)
        
        return f"📋 CLINICAL GUIDELINES: {topic}\n\n{result['answer']}"
    
    
    @tool
    def calculate_medical_score(score_name: str, patient_data: str) -> str:
        """
        Calculate medical scores (APGAR, Glasgow Coma Scale, CURB-65, etc.)
        Use this for standardized clinical assessments.
        
        Args:
            score_name: Name of the score (e.g., "Glasgow Coma Scale", "APGAR")
            patient_data: Patient data needed for calculation
        
        Returns:
            Calculated score with interpretation
        """
        
        query = f"how to calculate {score_name} interpretation with patient data: {patient_data}"
        
        result = rag_system.answer_with_sources(query, k=2)
        
        return f"📊 {score_name.upper()}\n\n{result['answer']}"
    
    
    @tool
    def get_pediatric_guidance(condition: str, age_months: int) -> str:
        """
        Get pediatric-specific guidance including age-appropriate treatments.
        Use this for patients under 18 years old.
        
        Args:
            condition: Pediatric condition or symptom
            age_months: Child's age in months
        
        Returns:
            Pediatric treatment guidance
        """
        
        age_years = age_months / 12
        query = f"pediatric management of {condition} in {age_years:.1f} year old child age-appropriate treatment"
        
        result = rag_system.answer_with_sources(query, k=3)
        
        return f"👶 PEDIATRIC GUIDANCE\n\nAge: {age_months} months ({age_years:.1f} years)\nCondition: {condition}\n\n{result['answer']}"
    
    
    @tool
    def search_infectious_disease_info(disease: str) -> str:
        """
        Look up infectious disease information including transmission, treatment, and prevention.
        Use this for suspected or confirmed infectious diseases.
        
        Args:
            disease: Name of infectious disease
        
        Returns:
            Disease information with treatment and prevention measures
        """
        
        query = f"{disease} transmission symptoms treatment prevention infection control"
        
        result = rag_system.answer_with_sources(query, k=4)
        
        return f"🦠 INFECTIOUS DISEASE: {disease.upper()}\n\n{result['answer']}"
    
    
    @tool
    def get_maternal_health_guidance(condition: str, gestational_age: int = None) -> str:
        """
        Get guidance for maternal health issues during pregnancy or postpartum.
        Use this for pregnant or postpartum patients.
        
        Args:
            condition: Maternal health condition or concern
            gestational_age: Weeks of gestation if applicable
        
        Returns:
            Maternal health guidance
        """
        
        gestation_context = f" at {gestational_age} weeks gestation" if gestational_age else ""
        query = f"maternal health management of {condition}{gestation_context} pregnancy obstetric care"
        
        result = rag_system.answer_with_sources(query, k=3)
        
        return f"🤰 MATERNAL HEALTH\n\nCondition: {condition}\n{f'Gestational Age: {gestational_age} weeks' if gestational_age else ''}\n\n{result['answer']}"
    
    
    @tool
    def lookup_lab_interpretation(test_name: str, result_value: str) -> str:
        """
        Interpret laboratory test results.
        Use this to understand what lab values mean clinically.
        
        Args:
            test_name: Name of laboratory test
            result_value: Test result value with units
        
        Returns:
            Clinical interpretation of the test result
        """
        
        query = f"clinical interpretation of {test_name} result {result_value} what it means clinical significance"
        
        result = rag_system.answer_with_sources(query, k=2)
        
        return f"🔬 LAB INTERPRETATION\n\nTest: {test_name}\nResult: {result_value}\n\n{result['answer']}"
    
    
    # Return all tools
    return [
        search_medical_knowledge,
        get_emergency_protocol,
        lookup_drug_information,
        get_diagnostic_guidance,
        search_clinical_guidelines,
        calculate_medical_score,
        get_pediatric_guidance,
        search_infectious_disease_info,
        get_maternal_health_guidance,
        lookup_lab_interpretation
    ]
