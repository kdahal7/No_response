#!/usr/bin/env python3
"""
Verbose app test to catch startup errors
"""

import traceback
import sys

print("🚀 Starting verbose app test...")

try:
    print("1️⃣ Importing modules...")
    
    import os
    from dotenv import load_dotenv
    print("✅ Basic imports OK")
    
    load_dotenv()
    print("✅ Environment loaded")
    
    from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    print("✅ FastAPI imports OK")
    
    import requests
    import tempfile
    from typing import List, Dict, Any, Optional
    import hashlib
    import asyncio
    import time
    from concurrent.futures import ThreadPoolExecutor
    import logging
    print("✅ Standard library imports OK")
    
    from extract import extract_text_from_pdf
    from search import DocumentProcessor, SemanticSearch
    from local_llm_processor import LLMProcessor
    from decision_engine import DecisionEngine
    print("✅ Custom module imports OK")
    
    print("\n2️⃣ Initializing components...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("✅ Logging configured")
    
    app = FastAPI(title="LLM-Powered Query-Retrieval System", version="1.0.0")
    print("✅ FastAPI app created")
    
    # Global instances with caching
    print("Creating DocumentProcessor...")
    document_processor = DocumentProcessor()
    print("✅ DocumentProcessor created")
    
    print("Creating SemanticSearch...")
    semantic_search = SemanticSearch()
    print("✅ SemanticSearch created")
    
    print("Creating LLMProcessor...")
    llm_processor = LLMProcessor()
    print("✅ LLMProcessor created")
    
    print("Creating DecisionEngine...")
    decision_engine = DecisionEngine(llm_processor)
    print("✅ DecisionEngine created")
    
    # In-memory cache for processed documents
    DOCUMENT_CACHE = {}
    EMBEDDING_CACHE = {}
    print("✅ Caches initialized")
    
    # Thread pool for parallel processing
    executor = ThreadPoolExecutor(max_workers=4)
    print("✅ Thread pool created")
    
    print("\n3️⃣ Testing component initialization...")
    
    # Test semantic search model loading
    try:
        test_embedding = semantic_search.model.encode(["test"], show_progress_bar=False)
        print("✅ Semantic search model working")
    except Exception as e:
        print(f"⚠️  Semantic search model issue: {e}")
    
    # Test LLM processor
    try:
        test_query = llm_processor.parse_query("What is the waiting period?")
        print("✅ LLM processor working")
    except Exception as e:
        print(f"⚠️  LLM processor issue: {e}")
    
    print("\n4️⃣ Defining request/response models...")
    
    class QueryRequest(BaseModel):
        documents: str
        questions: List[str]

    class QueryResponse(BaseModel):
        answers: List[str]
    
    print("✅ Models defined")
    
    print("\n5️⃣ Adding simple health check endpoint...")
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "LLM Query-Retrieval System is running"}
    
    print("✅ Health endpoint added")
    
    print("\n6️⃣ Starting server...")
    
    if __name__ == "__main__":
        import uvicorn
        print("Starting server on http://localhost:8000...")
        print("Health check: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Missing dependency or module not found")
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
    
print("Test completed.")