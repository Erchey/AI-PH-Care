"""
Configuration Module for Healthcare AI Agent
Manages environment variables, API keys, and settings
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Config:
    """Configuration for Healthcare AI Agent"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    # Alternative models: "mixtral-8x7b-32768", "llama3-8b-8192"
    
    # Multilingual Support
    SUPPORTED_LANGUAGES = [
        "English", "Spanish", "French", "Portuguese", 
        "Swahili", "Hindi", "Arabic", "Mandarin", 'Nigerian Pidgin'
    ]
    
    # Triage Priority Levels
    TRIAGE_LEVELS = {
        "CRITICAL": {"color": "red", "max_wait_minutes": 0},
        "URGENT": {"color": "orange", "max_wait_minutes": 15},
        "SEMI_URGENT": {"color": "yellow", "max_wait_minutes": 60},
        "NON_URGENT": {"color": "green", "max_wait_minutes": 120}
    }
    
    # Vital Signs Thresholds (Adult)
    VITAL_SIGNS_NORMAL = {
        "temperature": {"min": 36.1, "max": 37.2, "unit": "°C"},
        "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
        "respiratory_rate": {"min": 12, "max": 20, "unit": "breaths/min"},
        "blood_pressure_systolic": {"min": 90, "max": 140, "unit": "mmHg"},
        "blood_pressure_diastolic": {"min": 60, "max": 90, "unit": "mmHg"},
        "oxygen_saturation": {"min": 95, "max": 100, "unit": "%"}
    }
    
    # Emergency Keywords (Multilingual)
    EMERGENCY_KEYWORDS = {
        "en": ["chest pain", "difficulty breathing", "unconscious", "severe bleeding", 
               "stroke", "seizure", "suicide", "overdose"],
        "es": ["dolor de pecho", "dificultad para respirar", "inconsciente", 
               "sangrado severo", "derrame cerebral"],
        "fr": ["douleur thoracique", "difficulté à respirer", "inconscient", 
               "saignement sévère", "accident vasculaire"],
        "sw": ["maumivu ya kifua", "ugumu wa kupumua", "amezimia", "kutokwa damu kwingi"]
    }
    
    # Referral Criteria
    REFERRAL_SPECIALTIES = [
        "cardiology", "neurology", "pediatrics", "obstetrics",
        "surgery", "psychiatry", "orthopedics", "oncology"
    ]
    
    # Resource Thresholds
    RESOURCE_ALERTS = {
        "bed_occupancy": 80,  # Percentage
        "staff_ratio": 20,    # Patients per staff
        "medication_stock": 20  # Percentage remaining
    }
    
    # Offline Mode Settings
    OFFLINE_CACHE_DIR = Path("./cache")
    OFFLINE_MODELS_DIR = Path("./models")
    ENABLE_OFFLINE_MODE = True
    
    # Database Configuration (if using local storage)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///healthcare_agent.db")
    
    # API Endpoints (for integration)
    HEALTH_FACILITY_API = os.getenv("FACILITY_API_URL", "")
    DRUG_DATABASE_API = os.getenv("DRUG_API_URL", "")
    PATIENT_RECORDS_API = os.getenv("PATIENT_API_URL", "")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "healthcare_agent.log"
    
    # Session Management
    SESSION_TIMEOUT = 3600  # seconds
    MAX_CONVERSATION_LENGTH = 50  # messages
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def get_language_config(cls, language: str):
        """Get language-specific configuration"""
        return {
            "language": language,
            "emergency_keywords": cls.EMERGENCY_KEYWORDS.get(
                language[:2].lower(), 
                cls.EMERGENCY_KEYWORDS["en"]
            ),
            "supported": language in cls.SUPPORTED_LANGUAGES
        }


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"Warning: {e}")
    print("Set GROQ_API_KEY in your .env file")
