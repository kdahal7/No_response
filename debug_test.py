#!/usr/bin/env python3
"""
Debug script to test each component individually
"""

print("🔍 Starting debug tests...")

# Test 1: Basic imports
print("\n1️⃣ Testing basic imports...")
try:
    import os
    import sys
    print("✅ Basic imports OK")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

# Test 2: FastAPI imports
print("\n2️⃣ Testing FastAPI imports...")
try:
    from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    print("✅ FastAPI imports OK")
except Exception as e:
    print(f"❌ FastAPI imports failed: {e}")
    exit(1)

# Test 3: Document processing imports
print("\n3️⃣ Testing document processing imports...")
try:
    from extract import extract_text_from_pdf
    print("✅ Extract module OK")
except Exception as e:
    print(f"❌ Extract module failed: {e}")

try:
    from search import DocumentProcessor, SemanticSearch
    print("✅ Search module OK")
except Exception as e:
    print(f"❌ Search module failed: {e}")

# Test 4: LLM processor (this is likely where the issue is)
print("\n4️⃣ Testing LLM processor import...")
try:
    from local_llm_processor import LLMProcessor
    print("✅ LLM processor import OK")
except Exception as e:
    print(f"❌ LLM processor import failed: {e}")
    print(f"Error details: {type(e).__name__}: {e}")

# Test 5: Decision engine
print("\n5️⃣ Testing decision engine import...")
try:
    from decision_engine import DecisionEngine
    print("✅ Decision engine import OK")
except Exception as e:
    print(f"❌ Decision engine import failed: {e}")

# Test 6: Ollama availability
print("\n6️⃣ Testing Ollama availability...")
try:
    import ollama
    print("✅ Ollama package available")
    
    # Test if Ollama service is running
    try:
        models = ollama.list()
        print(f"✅ Ollama service running. Available models: {len(models.get('models', []))}")
        
        # Check for Gemma model
        model_names = [m['name'] for m in models.get('models', [])]
        if any('gemma' in name.lower() for name in model_names):
            print("✅ Gemma model found")
        else:
            print("⚠️  No Gemma model found. Available models:", model_names)
            
    except Exception as e:
        print(f"⚠️  Ollama service not running: {e}")
        
except Exception as e:
    print(f"❌ Ollama not available: {e}")

# Test 7: Try to create LLM processor instance
print("\n7️⃣ Testing LLM processor initialization...")
try:
    from local_llm_processor import LLMProcessor
    processor = LLMProcessor()
    print("✅ LLM processor created successfully")
except Exception as e:
    print(f"❌ LLM processor initialization failed: {e}")
    print(f"Error details: {type(e).__name__}: {e}")

print("\n✅ Debug tests completed!")
print("\n💡 Next steps:")
print("1. Fix any ❌ errors shown above")
print("2. If Ollama service isn't running, start it: 'ollama serve'")
print("3. If no Gemma model, install it: 'ollama pull gemma3:12b'")
print("4. Try running app.py again")