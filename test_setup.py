#!/usr/bin/env python3
"""
Utility script to test the RAG system setup and dependencies
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} is too old. Please use Python 3.8+")
        return False

def check_imports():
    """Check if all required packages are installed"""
    required_packages = [
        ("streamlit", "Streamlit"),
        ("dotenv", "python-dotenv"),
        ("pdfplumber", "PDFPlumber"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("openai", "OpenAI"),
    ]
    
    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is not installed")
            all_installed = False
    
    return all_installed

def check_env_file():
    """Check if .env file exists and has OpenAI API key"""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file exists")
        
        # Check for API key
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv("OPENAI_API_KEY"):
            print("✅ OpenAI API key is configured")
            return True
        else:
            print("⚠️  OpenAI API key not found in .env (optional but recommended)")
            return True
    else:
        print("⚠️  .env file not found (optional)")
        return True

def check_folders():
    """Check if required folders exist"""
    folders_to_check = [
        ("legal_docs", "Document folder"),
    ]
    
    all_exist = True
    for folder, name in folders_to_check:
        folder_path = Path(folder)
        if folder_path.exists() and folder_path.is_dir():
            pdf_files = list(folder_path.glob("*.pdf"))
            print(f"✅ {name} exists ({len(pdf_files)} PDF files found)")
        else:
            print(f"⚠️  {name} '{folder}' not found (will be created if needed)")
    
    return all_exist

def test_basic_functionality():
    """Test basic functionality of the system"""
    print("\n📋 Testing basic functionality...")
    
    try:
        # Test embedding model loading
        from sentence_transformers import SentenceTransformer
        print("⏳ Loading embedding model (this may take a moment on first run)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test embedding generation
        test_text = "This is a test legal document."
        embedding = model.encode(test_text)
        print(f"✅ Embedding model works (generated {len(embedding)}-dimensional vector)")
        
        return True
    except Exception as e:
        print(f"❌ Error testing functionality: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🔍 RAG System Setup Checker")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("Environment Setup", check_env_file),
        ("Folder Structure", check_folders),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📌 Checking {name}...")
        print("-" * 40)
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Your system is ready.")
        print("\nRun the application with:")
        print("  streamlit run rag_rerank_app.py")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()