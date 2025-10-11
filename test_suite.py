"""
Test Suite for Healthcare AI Agent
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTriageModule:
    """Test AI Triage Module"""
    
    def test_critical_patient_assessment(self):
        """Test that critical patients are correctly identified"""
        from modules.triage import AITriageModule
        
        # Mock LLM and RAG
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="CRITICAL priority due to severe chest pain and vital signs"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        triage = AITriageModule(mock_llm, mock_rag)
        
        patient = {
            "age": 58,
            "chief_complaint": "severe chest pain",
            "symptoms": ["chest pain", "shortness of breath"],
            "vital_signs": {"oxygen_saturation": 88, "heart_rate": 120}
        }
        
        result = triage.assess_patient(patient, "Assess urgency")
        
        assert result['priority'] == "CRITICAL"
        assert result['requires_immediate_doctor'] == True
    
    def test_non_urgent_patient(self):
        """Test that stable patients get appropriate priority"""
        from modules.triage import AITriageModule
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="NON-URGENT - stable vital signs, routine care appropriate"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        triage = AITriageModule(mock_llm, mock_rag)
        
        patient = {
            "age": 25,
            "chief_complaint": "minor cold symptoms",
            "symptoms": ["runny nose", "mild cough"],
            "vital_signs": {"temperature": 37.5}
        }
        
        result = triage.assess_patient(patient, "Assess patient")
        
        assert result['priority'] == "NON_URGENT"
        assert result['estimated_wait_minutes'] == 120


class TestReferralModule:
    """Test AI Referral Module"""
    
    def test_specialist_referral_needed(self):
        """Test that appropriate referrals are recommended"""
        from modules.referral import AIReferralModule
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="URGENT REFERRAL to cardiology needed immediately"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        referral = AIReferralModule(mock_llm, mock_rag)
        
        patient = {
            "age": 60,
            "chief_complaint": "chest pain",
            "symptoms": ["chest pain", "palpitations"]
        }
        
        result = referral.generate_referral(patient, {}, "English")
        
        assert result['referral_decision'] in ["URGENT_REFERRAL", "IMMEDIATE_TRANSFER"]
        assert result['requires_immediate_transfer'] or result['urgency_level'] == "URGENT"
    
    def test_phc_management_appropriate(self):
        """Test that PHC-manageable cases don't get unnecessary referral"""
        from modules.referral import AIReferralModule
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="MANAGE AT PHC level - can be handled with available resources"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        referral = AIReferralModule(mock_llm, mock_rag)
        
        patient = {
            "age": 30,
            "chief_complaint": "mild upper respiratory infection",
            "symptoms": ["cough", "sore throat"]
        }
        
        result = referral.generate_referral(patient, {}, "English")
        
        assert result['referral_decision'] == "MANAGE_AT_PHC"


class TestResourceModule:
    """Test Resource Allocation Module"""
    
    def test_resource_alert_generation(self):
        """Test that resource alerts are correctly generated"""
        from modules.resource_allocation import AIResourceModule
        
        mock_llm = Mock()
        mock_rag = Mock()
        
        resource = AIResourceModule(mock_llm, mock_rag)
        
        facility = {
            "beds": {"total": 10, "occupied": 10},
            "staff": {"on_duty": 2},
            "patients": {"total": 40}
        }
        
        alerts = resource._identify_resource_alerts(facility)
        
        assert len(alerts) > 0
        assert any("bed" in alert.lower() or "staff" in alert.lower() for alert in alerts)
    
    def test_resource_optimization(self):
        """Test resource optimization recommendations"""
        from modules.resource_allocation import AIResourceModule
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Immediate actions: 1. Prepare discharge for stable patients 2. Activate surge protocol"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        resource = AIResourceModule(mock_llm, mock_rag)
        
        facility = {
            "beds": {"total": 10, "occupied": 9},
            "staff": {"on_duty": 3}
        }
        
        result = resource.optimize_resources(facility, {}, "English")
        
        assert "recommendations" in result
        assert len(result['recommendations']) > 0


class TestRoutingModule:
    """Test Patient Routing Module"""
    
    def test_patient_routing(self):
        """Test patient routing recommendations"""
        from modules.patient_routing import AIRoutingModule
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Route to emergency nurse for immediate assessment, then doctor"))
        mock_rag = Mock()
        mock_rag.retrieve = Mock(return_value=[])
        
        routing = AIRoutingModule(mock_llm, mock_rag)
        
        patient = {
            "age": 40,
            "chief_complaint": "moderate injury",
            "symptoms": ["pain", "swelling"]
        }
        
        result = routing.route_patient(patient, {}, "English")
        
        assert "routing_plan" in result
        assert result['recommended_provider'] in ["doctor", "nurse", "clinical_officer", "specialist", "emergency"]


class TestRAGSystem:
    """Test RAG System"""
    
    def test_document_ingestion(self):
        """Test that documents can be ingested"""
        from modules.rag_system import MedicalRAGSystem
        
        mock_llm = Mock()
        
        # This is a basic test - in production you'd test with actual documents
        rag = MedicalRAGSystem(mock_llm)
        
        assert rag.vector_store is not None
        assert rag.embeddings is not None
    
    def test_retrieval(self):
        """Test document retrieval"""
        from modules.rag_system import MedicalRAGSystem
        
        mock_llm = Mock()
        rag = MedicalRAGSystem(mock_llm)
        
        # Basic retrieval test
        # Note: Will return empty if no documents ingested
        results = rag.retrieve("test query", k=2)
        
        assert isinstance(results, list)


class TestTools:
    """Test Agent Tools"""
    
    def test_tools_initialization(self):
        """Test that tools can be initialized"""
        from modules.rag_system import MedicalRAGSystem
        from tools import get_all_tools
        
        mock_llm = Mock()
        rag = MedicalRAGSystem(mock_llm)
        
        tools = get_all_tools(rag)
        
        assert len(tools) > 0
        assert all(hasattr(tool, 'name') for tool in tools)


class TestAgent:
    """Test Main Healthcare Agent"""
    
    @patch('agent.ChatGroq')
    def test_agent_initialization(self, mock_groq):
        """Test agent can be initialized"""
        from agent import HealthcareAgent
        
        # Mock the LLM
        mock_groq.return_value = Mock()
        
        try:
            agent = HealthcareAgent()
            assert agent is not None
            assert agent.graph is not None
        except Exception as e:
            # If it fails due to API key, that's expected in test environment
            if "GROQ_API_KEY" in str(e):
                pytest.skip("Groq API key not configured for testing")
            else:
                raise
    
    @patch('agent.ChatGroq')
    def test_agent_run_method(self, mock_groq):
        """Test agent run method"""
        from agent import HealthcareAgent
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Assessment complete"))
        mock_groq.return_value = mock_llm
        
        try:
            agent = HealthcareAgent()
            
            patient = {
                "age": 30,
                "chief_complaint": "test complaint"
            }
            
            result = agent.run(
                user_input="Test query",
                patient_data=patient
            )
            
            assert "messages" in result
        except Exception as e:
            if "GROQ_API_KEY" in str(e):
                pytest.skip("Groq API key not configured")
            else:
                raise


class TestConfig:
    """Test Configuration"""
    
    def test_config_loading(self):
        """Test configuration loads correctly"""
        from config import Config
        
        config = Config()
        
        assert hasattr(config, 'GROQ_MODEL')
        assert hasattr(config, 'SUPPORTED_LANGUAGES')
        assert hasattr(config, 'TRIAGE_LEVELS')
        
        # Test supported languages
        assert 'English' in config.SUPPORTED_LANGUAGES
        assert len(config.SUPPORTED_LANGUAGES) >= 8
    
    def test_triage_levels(self):
        """Test triage level configuration"""
        from config import Config
        
        config = Config()
        
        assert 'CRITICAL' in config.TRIAGE_LEVELS
        assert 'URGENT' in config.TRIAGE_LEVELS
        assert 'SEMI_URGENT' in config.TRIAGE_LEVELS
        assert 'NON_URGENT' in config.TRIAGE_LEVELS


class TestIntegration:
    """Integration Tests"""
    
    @pytest.mark.integration
    @patch('agent.ChatGroq')
    def test_full_triage_workflow(self, mock_groq):
        """Test complete triage workflow"""
        from agent import HealthcareAgent
        
        mock_llm = Mock()
        
        # Mock classifier response
        mock_llm.invoke = Mock(side_effect=[
            Mock(content="triage"),  # Classification
            Mock(content="CRITICAL priority - immediate medical attention required"),  # Triage
            Mock(content="Assessment complete", tool_calls=None)  # Final response
        ])
        
        mock_groq.return_value = mock_llm
        
        try:
            agent = HealthcareAgent()
            
            patient = {
                "age": 65,
                "chief_complaint": "chest pain",
                "vital_signs": {"oxygen_saturation": 90}
            }
            
            result = agent.run(
                user_input="Perform triage on this patient",
                patient_data=patient
            )
            
            assert result is not None
            assert "messages" in result
            
        except Exception as e:
            if "GROQ_API_KEY" in str(e):
                pytest.skip("Groq API key not configured")
            else:
                raise


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
