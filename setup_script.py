"""
Setup Script for Healthcare AI Agent
Initializes the project structure and dependencies
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create necessary directories"""
    
    directories = [
        "modules",
        "medical_docs",
        "medical_docs/protocols",
        "medical_docs/guidelines",
        "medical_docs/formularies",
        "vector_store",
        "cache",
        "logs",
        "tests"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created: {directory}/")
    
    # Create __init__.py files
    init_files = ["modules/__init__.py", "tests/__init__.py"]
    for init_file in init_files:
        Path(init_file).touch()
        print(f"   ✓ Created: {init_file}")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("\n🔧 Creating .env file...")
            with open(".env.example", "r") as example:
                content = example.read()
            with open(".env", "w") as env:
                env.write(content)
            print("   ✓ Created .env file")
            print("   ⚠️  Remember to add your GROQ_API_KEY in .env")
        else:
            print("\n⚠️  .env.example not found. Creating minimal .env...")
            with open(".env", "w") as env:
                env.write("GROQ_API_KEY=your_api_key_here\n")
                env.write("GROQ_MODEL=llama-3.1-70b-versatile\n")
            print("   ✓ Created .env file")
    else:
        print("\n✓ .env file already exists")


def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
vector_store/
cache/
logs/
*.log
medical_docs/*.pdf
medical_docs/*.docx

# Keep directories with .gitkeep
!medical_docs/protocols/.gitkeep
!medical_docs/guidelines/.gitkeep
!medical_docs/formularies/.gitkeep
"""
    
    if not Path(".gitignore").exists():
        print("\n📝 Creating .gitignore...")
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("   ✓ Created .gitignore")
    else:
        print("\n✓ .gitignore already exists")


def create_gitkeep_files():
    """Create .gitkeep files to preserve empty directories in git"""
    
    gitkeep_dirs = [
        "medical_docs/protocols",
        "medical_docs/guidelines", 
        "medical_docs/formularies",
        "logs"
    ]
    
    print("\n📌 Creating .gitkeep files...")
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()
        print(f"   ✓ Created: {gitkeep_path}")


def create_sample_medical_doc():
    """Create a sample medical document for testing"""
    
    sample_content = """# Sample Medical Protocol - Fever Management in Adults

## Assessment
1. Take complete vital signs
2. Measure temperature accurately
3. Assess for signs of serious infection

## Fever Thresholds
- Normal: 36.1-37.2°C (97-99°F)
- Low-grade fever: 37.3-38.0°C (99-100.4°F)
- Fever: 38.1-39.0°C (100.5-102.2°F)
- High fever: >39.0°C (>102.2°F)

## Red Flags (Immediate Medical Attention)
- Temperature >40°C (104°F)
- Altered consciousness
- Severe headache with neck stiffness
- Difficulty breathing
- Chest pain
- Persistent vomiting
- Signs of dehydration

## Management
### Mild Fever (38.1-38.5°C)
- Rest and hydration
- Monitor temperature
- Paracetamol 500-1000mg every 4-6 hours (max 4g/day)

### Moderate to High Fever (>38.5°C)
- Aggressive cooling measures
- Paracetamol or Ibuprofen
- Investigate cause
- Consider blood tests

## When to Refer
- Fever >39.5°C not responding to treatment
- Immunocompromised patients
- Suspected serious bacterial infection
- Persistent fever >3 days
- Patient deteriorating

## Follow-up
- Review in 24-48 hours if not improving
- Return immediately if red flags develop
"""
    
    sample_path = Path("medical_docs/protocols/sample_fever_protocol.md")
    if not sample_path.exists():
        print("\n📄 Creating sample medical document...")
        with open(sample_path, "w") as f:
            f.write(sample_content)
        print(f"   ✓ Created: {sample_path}")
        print("   📚 This sample document will be used for RAG demonstration")


def check_dependencies():
    """Check if required packages are installed"""
    
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "langgraph",
        "langchain",
        "langchain_groq",
        "chromadb",
        "sentence_transformers",
        "dotenv"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"   ✗ {package} (missing)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def display_next_steps():
    """Display next steps for the user"""
    
    print("\n" + "=" * 80)
    print("  🎉 Setup Complete!")
    print("=" * 80)
    
    print("\n📋 Next Steps:")
    print("\n1. Configure your Groq API key:")
    print("   - Get free API key: https://console.groq.com")
    print("   - Edit .env file and add: GROQ_API_KEY=your_key_here")
    
    print("\n2. Add medical documents (Optional but recommended):")
    print("   - Add PDFs to: medical_docs/protocols/")
    print("   - Add guidelines to: medical_docs/guidelines/")
    print("   - Add formularies to: medical_docs/formularies/")
    
    print("\n3. Install dependencies (if not already done):")
    print("   pip install -r requirements.txt")
    
    print("\n4. Run the demo:")
    print("   python demo.py")
    
    print("\n5. Start building:")
    print("   from agent import HealthcareAgent")
    print("   agent = HealthcareAgent()")
    print("   result = agent.run(user_input='Your query here')")
    
    print("\n📚 Resources:")
    print("   - README.md: Complete documentation")
    print("   - demo.py: Usage examples")
    print("   - config.py: Configuration options")
    
    print("\n" + "=" * 80)


def main():
    """Run setup"""
    
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║     🏥 Healthcare AI Agent - Setup                            ║")
    print("║     For Primary Healthcare Workers                            ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Create configuration files
        create_env_file()
        create_gitignore()
        create_gitkeep_files()
        
        # Create sample medical document
        create_sample_medical_doc()
        
        # Check dependencies
        deps_ok = check_dependencies()
        
        # Display next steps
        display_next_steps()
        
        if not deps_ok:
            print("\n⚠️  Warning: Some dependencies are missing.")
            print("   Install them before running the agent.")
        
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
