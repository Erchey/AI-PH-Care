"""
AI-Driven Resource Allocation Module
Optimizes healthcare resources using LLM reasoning
"""

from typing import Dict, List
import json


class AIResourceModule:
    """AI-powered resource allocation and optimization"""
    
    def __init__(self, llm, rag_system):
        self.llm = llm
        self.rag_system = rag_system
    
    def optimize_resources(
        self,
        facility_data: Dict,
        current_demand: Dict,
        language: str = "English"
    ) -> Dict:
        """
        Optimize resource allocation based on current facility state
        
        Args:
            facility_data: Current facility resources and constraints
            current_demand: Current patient demand and needs
            language: Preferred language
        
        Returns:
            Resource allocation recommendations
        """
        
        # Retrieve resource management guidelines
        guidelines = self._retrieve_resource_guidelines(facility_data, current_demand)
        
        # Build optimization prompt
        prompt = self._build_optimization_prompt(
            facility_data,
            current_demand,
            guidelines,
            language
        )
        
        # Get AI recommendations
        response = self.llm.invoke(prompt)
        
        # Structure response
        result = {
            "recommendations": response.content,
            "sources": guidelines,
            "priority_actions": self._extract_priority_actions(response.content),
            "alerts": self._identify_resource_alerts(facility_data)
        }
        
        return result
    
    def _retrieve_resource_guidelines(self, facility_data: Dict, demand: Dict) -> list:
        """Retrieve resource management guidelines using RAG"""
        
        query = """healthcare resource allocation optimization bed management 
                   staff scheduling patient flow resource-limited settings 
                   emergency resource allocation"""
        
        docs = self.rag_system.retrieve(query, k=3)
        return docs
    
    def _build_optimization_prompt(
        self,
        facility_data: Dict,
        current_demand: Dict,
        guidelines: list,
        language: str
    ) -> str:
        """Build resource optimization prompt"""
        
        guidelines_text = "\n\n".join([
            f"Guideline {i+1}:\n{doc['content']}"
            for i, doc in enumerate(guidelines)
        ]) if guidelines else "Using general resource management principles"
        
        prompt = f"""You are a healthcare resource management AI helping optimize resources at a Primary Healthcare facility.

CURRENT FACILITY STATUS:
{json.dumps(facility_data, indent=2)}

CURRENT DEMAND/PATIENT LOAD:
{json.dumps(current_demand, indent=2)}

RESOURCE MANAGEMENT GUIDELINES:
{guidelines_text}

YOUR TASK:
Provide intelligent resource allocation recommendations.

Analyze and recommend:

1. **BED ALLOCATION**:
   - Current bed utilization
   - Predicted capacity issues
   - Which patients to prioritize for beds
   - Discharge planning to free beds

2. **STAFF DEPLOYMENT**:
   - Optimal staff distribution
   - Areas needing more staff attention
   - Break scheduling to prevent burnout
   - Skill mix optimization

3. **EQUIPMENT PRIORITIZATION**:
   - Limited equipment allocation
   - Which patients need equipment most urgently
   - Equipment sharing strategies

4. **SUPPLY MANAGEMENT**:
   - Critical supplies running low
   - Conservation strategies
   - Alternative options if supplies depleted

5. **PATIENT FLOW OPTIMIZATION**:
   - Reduce bottlenecks
   - Fast-track stable patients
   - Efficient queue management

6. **EMERGENCY PREPAREDNESS**:
   - Reserve capacity for emergencies
   - Surge capacity plans
   - What to do if overwhelmed

7. **IMMEDIATE ACTIONS** (Top 3-5):
   - Most critical resource decisions needed now
   - Order by priority

8. **RISK MITIGATION**:
   - Potential resource crises
   - Prevention strategies

CONSTRAINTS TO CONSIDER:
- This is a resource-limited setting
- Staff may be overworked
- Limited budget and supplies
- Community health needs
- Cultural considerations

Language: {language}

Provide practical, actionable recommendations that can be implemented immediately.
"""
        
        return prompt
    
    def _extract_priority_actions(self, response: str) -> List[str]:
        """Extract priority actions from response"""
        
        # Simple extraction - look for numbered lists or bullet points
        actions = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for lines that appear to be action items
            if any(indicator in line.lower() for indicator in ['immediate', 'priority', 'urgent', 'critical']):
                if line.strip() and len(line.strip()) > 10:
                    actions.append(line.strip())
        
        return actions[:5]  # Return top 5
    
    def _identify_resource_alerts(self, facility_data: Dict) -> List[str]:
        """Identify critical resource alerts"""
        
        alerts = []
        
        # Check bed occupancy
        if 'beds' in facility_data:
            total_beds = facility_data['beds'].get('total', 0)
            occupied = facility_data['beds'].get('occupied', 0)
            if total_beds > 0:
                occupancy = (occupied / total_beds) * 100
                if occupancy > 95:
                    alerts.append(f"🚨 CRITICAL: Bed occupancy at {occupancy:.0f}%")
                elif occupancy > 85:
                    alerts.append(f"⚠️ WARNING: Bed occupancy at {occupancy:.0f}%")
        
        # Check staff ratios
        if 'staff' in facility_data and 'patients' in facility_data:
            staff_count = facility_data['staff'].get('on_duty', 0)
            patient_count = facility_data.get('patients', {}).get('total', 0)
            if staff_count > 0:
                ratio = patient_count / staff_count
                if ratio > 20:
                    alerts.append(f"🚨 HIGH STAFF LOAD: {ratio:.0f} patients per staff member")
        
        # Check supplies
        if 'supplies' in facility_data:
            for supply, data in facility_data['supplies'].items():
                if isinstance(data, dict) and 'level' in data:
                    level = data['level']
                    if level < 20:
                        alerts.append(f"⚠️ LOW SUPPLY: {supply} at {level}%")
        
        return alerts
    
    def allocate_limited_resource(
        self,
        resource_type: str,
        available_quantity: int,
        patients: List[Dict],
        language: str = "English"
    ) -> str:
        """
        Allocate a limited resource among multiple patients
        
        Args:
            resource_type: Type of resource (beds, oxygen, medications, etc.)
            available_quantity: How many units available
            patients: List of patients needing the resource
            language: Language preference
        
        Returns:
            Allocation recommendation with ethical justification
        """
        
        allocation_prompt = f"""You are making a difficult triage decision about limited healthcare resources.

RESOURCE: {resource_type}
AVAILABLE: {available_quantity} unit(s)

PATIENTS REQUESTING THIS RESOURCE:
{json.dumps(patients, indent=2)}

YOUR TASK:
Recommend fair and ethical allocation of this limited resource.

Provide:

1. **ALLOCATION DECISION**:
   - Which patient(s) should receive the resource
   - Clear ranking with justification

2. **ETHICAL REASONING**:
   - Principles guiding your decision (severity, likelihood of benefit, etc.)
   - Why this allocation is fair

3. **ALTERNATIVE CARE**:
   - What to do for patients who don't receive the resource
   - Alternative treatments or management

4. **DOCUMENTATION**:
   - How to document this difficult decision
   - Communication with patients/families

5. **ESCALATION**:
   - When to seek additional resources or transfers

ETHICAL PRINCIPLES TO CONSIDER:
- Medical urgency and severity
- Likelihood of benefit
- Fair chances (when severity similar)
- Transparency in decision-making
- Cultural sensitivity
- Legal and ethical guidelines

This is a sensitive situation requiring both clinical and ethical judgment.
Language: {language}

Provide a thoughtful, well-reasoned recommendation.
"""
        
        response = self.llm.invoke(allocation_prompt)
        return response.content
    
    def predict_resource_needs(
        self,
        historical_data: Dict,
        upcoming_factors: Dict,
        language: str = "English"
    ) -> str:
        """
        Predict future resource needs based on patterns
        
        Args:
            historical_data: Past utilization patterns
            upcoming_factors: Known future factors (season, events, etc.)
            language: Language preference
        
        Returns:
            Resource prediction and preparation recommendations
        """
        
        prediction_prompt = f"""Predict resource needs and help with planning.

HISTORICAL UTILIZATION DATA:
{json.dumps(historical_data, indent=2)}

UPCOMING FACTORS:
{json.dumps(upcoming_factors, indent=2)}

Provide:

1. **PREDICTED RESOURCE NEEDS** (next 1-7 days):
   - Expected patient volume
   - Likely resource requirements
   - Potential bottlenecks

2. **PREPARATION RECOMMENDATIONS**:
   - Resources to stock up
   - Staff scheduling adjustments
   - Equipment maintenance needed

3. **RISK SCENARIOS**:
   - Best case / Expected / Worst case
   - Contingency planning

4. **EARLY WARNING SIGNS**:
   - Indicators of resource strain
   - When to activate surge protocols

Language: {language}
"""
        
        response = self.llm.invoke(prediction_prompt)
        return response.content
    
    def optimize_patient_discharge(
        self,
        patients: List[Dict],
        bed_pressure: str,
        language: str = "English"
    ) -> str:
        """
        Recommend discharge planning to free resources
        
        Args:
            patients: Current admitted patients
            bed_pressure: Level of bed shortage (low/medium/high/critical)
            language: Language preference
        
        Returns:
            Discharge recommendations
        """
        
        discharge_prompt = f"""Help optimize discharge planning to free up resources safely.

CURRENT PATIENTS:
{json.dumps(patients, indent=2)}

BED PRESSURE: {bed_pressure}

Provide:

1. **DISCHARGE CANDIDATES**:
   - Patients safe for discharge
   - Priority order
   - Reasoning for each

2. **DISCHARGE CRITERIA MET**:
   - Clinical stability indicators
   - Home care feasibility

3. **DISCHARGE PLANNING**:
   - Medications to send home
   - Follow-up arrangements
   - Patient education needed

4. **PATIENTS TO KEEP**:
   - Why they're not ready for discharge
   - Expected length of stay

5. **STEP-DOWN OPTIONS**:
   - Patients who could move to observation
   - Community-based care alternatives

IMPORTANT:
- Patient safety is paramount
- Only recommend discharge if clinically appropriate
- Consider social factors (home support, transportation)

Language: {language}
"""
        
        response = self.llm.invoke(discharge_prompt)
        return response.content
