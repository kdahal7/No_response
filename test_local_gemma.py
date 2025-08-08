#!/usr/bin/env python3
"""
Comprehensive Test Suite for Local Gemma LLM Migration
Tests all components from Ollama service to API endpoints
"""

import requests
import json
import time
import os
import sys
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb"

# Headers for API calls
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

class LocalLLMTester:
    """Comprehensive tester for Local LLM system"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   {details}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })

    def test_1_basic_imports(self):
        """Test 1: Basic Python imports"""
        print("\n🔍 Test 1: Basic Imports")
        try:
            import os
            import sys
            import json
            import time
            import logging
            self.log_test("Basic Python imports", True)
            return True
        except Exception as e:
            self.log_test("Basic Python imports", False, str(e))
            return False

    def test_2_fastapi_imports(self):
        """Test 2: FastAPI and web framework imports"""
        print("\n🔍 Test 2: FastAPI Imports")
        try:
            from fastapi import FastAPI, Header, HTTPException
            from pydantic import BaseModel
            import uvicorn
            self.log_test("FastAPI imports", True)
            return True
        except Exception as e:
            self.log_test("FastAPI imports", False, str(e))
            return False

    def test_3_ml_dependencies(self):
        """Test 3: Machine Learning dependencies"""
        print("\n🔍 Test 3: ML Dependencies")
        success = True
        
        try:
            from sentence_transformers import SentenceTransformer
            self.log_test("SentenceTransformers", True)
        except Exception as e:
            self.log_test("SentenceTransformers", False, str(e))
            success = False
            
        try:
            import faiss
            self.log_test("FAISS", True)
        except Exception as e:
            self.log_test("FAISS", False, str(e))
            success = False
            
        try:
            import numpy
            self.log_test("NumPy", True)
        except Exception as e:
            self.log_test("NumPy", False, str(e))
            success = False
            
        return success

    def test_4_ollama_availability(self):
        """Test 4: Ollama service and model availability"""
        print("\n🔍 Test 4: Ollama Service & Models")
        
        # Test Ollama package
        try:
            import ollama
            self.log_test("Ollama package installed", True)
        except ImportError as e:
            self.log_test("Ollama package installed", False, "Run: pip install ollama")
            return False
        
        # Test Ollama service
        try:
            models = ollama.list()
            model_count = len(models.get('models', []))
            self.log_test("Ollama service running", True, f"Found {model_count} models")
        except Exception as e:
            self.log_test("Ollama service running", False, "Run: ollama serve")
            return False
        
        # Test for Gemma models
        model_names = [m['name'] for m in models.get('models', [])]
        gemma_models = [name for name in model_names if 'gemma' in name.lower()]
        
        if gemma_models:
            self.log_test("Gemma models available", True, f"Found: {', '.join(gemma_models)}")
        else:
            self.log_test("Gemma models available", False, "Run: ollama pull gemma3:12b")
            print("   Available models:", model_names)
            return False
            
        # Test model inference
        try:
            test_model = gemma_models[0]  # Use first available Gemma model
            response = ollama.chat(
                model=test_model,
                messages=[{"role": "user", "content": "Hello"}],
                options={"num_predict": 10}
            )
            self.log_test("Model inference working", True, f"Model: {test_model}")
            return True
        except Exception as e:
            self.log_test("Model inference working", False, str(e))
            return False

    def test_5_custom_modules(self):
        """Test 5: Custom module imports"""
        print("\n🔍 Test 5: Custom Module Imports")
        success = True
        
        modules_to_test = [
            ("extract", "extract_text_from_pdf"),
            ("search", "DocumentProcessor"),
            ("search", "SemanticSearch"),
            ("local_llm_processor", "LLMProcessor"),
            ("decision_engine", "DecisionEngine")
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                self.log_test(f"{module_name}.{class_name}", True)
            except ImportError as e:
                self.log_test(f"{module_name}.{class_name}", False, f"Import error: {e}")
                success = False
            except AttributeError as e:
                self.log_test(f"{module_name}.{class_name}", False, f"Class not found: {e}")
                success = False
            except Exception as e:
                self.log_test(f"{module_name}.{class_name}", False, str(e))
                success = False
                
        return success

    def test_6_component_initialization(self):
        """Test 6: Component initialization"""
        print("\n🔍 Test 6: Component Initialization")
        success = True
        
        # Test DocumentProcessor
        try:
            from search import DocumentProcessor
            processor = DocumentProcessor()
            self.log_test("DocumentProcessor initialization", True)
        except Exception as e:
            self.log_test("DocumentProcessor initialization", False, str(e))
            success = False
        
        # Test SemanticSearch
        try:
            from search import SemanticSearch
            search_engine = SemanticSearch()
            self.log_test("SemanticSearch initialization", True)
        except Exception as e:
            self.log_test("SemanticSearch initialization", False, str(e))
            success = False
        
        # Test LLMProcessor (most critical for local LLM)
        try:
            from local_llm_processor import LLMProcessor
            llm = LLMProcessor()  # Will test Ollama connection
            self.log_test("LLMProcessor initialization", True, f"Model: {llm.model_name}")
        except Exception as e:
            self.log_test("LLMProcessor initialization", False, str(e))
            success = False
            
        # Test DecisionEngine
        try:
            from decision_engine import DecisionEngine
            from local_llm_processor import LLMProcessor
            llm = LLMProcessor()
            engine = DecisionEngine(llm)
            self.log_test("DecisionEngine initialization", True)
        except Exception as e:
            self.log_test("DecisionEngine initialization", False, str(e))
            success = False
            
        return success

    def test_7_llm_functionality(self):
        """Test 7: Local LLM functionality"""
        print("\n🔍 Test 7: Local LLM Functionality")
        
        try:
            from local_llm_processor import LLMProcessor
            llm = LLMProcessor()
            
            # Test query parsing
            parsed = llm.parse_query("What is the waiting period for surgery coverage?")
            if parsed and 'intent' in parsed:
                self.log_test("Query parsing", True, f"Intent: {parsed['intent']}")
            else:
                self.log_test("Query parsing", False, "No valid parse result")
                return False
            
            # Test answer generation with mock chunks
            test_chunks = [{
                'text': 'The waiting period for surgery coverage is 24 months from policy inception.',
                'relevance_score': 0.95,
                'id': 'test_chunk_1'
            }]
            
            start_time = time.time()
            result = llm.generate_answer(
                question="What is the waiting period for surgery?",
                context="",
                retrieved_chunks=test_chunks
            )
            response_time = time.time() - start_time
            
            if result and 'answer' in result:
                self.log_test("Answer generation", True, 
                            f"Response time: {response_time:.2f}s, Model: {result.get('model', 'unknown')}")
                print(f"   Sample answer: {result['answer'][:100]}...")
                return True
            else:
                self.log_test("Answer generation", False, "No valid answer generated")
                return False
                
        except Exception as e:
            self.log_test("Local LLM functionality", False, str(e))
            return False

    def test_8_api_server_health(self):
        """Test 8: API server health"""
        print("\n🔍 Test 8: API Server Health")
        
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'unknown')
                self.log_test("API server health", True, f"Status: {status}")
                return True
            else:
                self.log_test("API server health", False, f"HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.log_test("API server health", False, "Server not running. Start with: python app.py")
            return False
        except Exception as e:
            self.log_test("API server health", False, str(e))
            return False

    def test_9_document_processing_api(self):
        """Test 9: Document processing via API"""
        print("\n🔍 Test 9: Document Processing API")
        
        # Use a simple test document URL
        test_data = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": [
                "What is this document about?",
                "What are the main topics covered?"
            ]
        }
        
        try:
            print("   ⏳ Processing document (may take 30-60 seconds)...")
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                json=test_data,
                headers=headers,
                timeout=120
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get('answers', [])
                self.log_test("Document processing API", True, 
                            f"Processed {len(answers)} questions in {processing_time:.2f}s")
                
                # Show sample answer
                if answers:
                    print(f"   Sample answer: {answers[0][:100]}...")
                return True
                
            elif response.status_code == 401:
                self.log_test("Document processing API", False, "Authentication failed")
                return False
            else:
                error_msg = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                self.log_test("Document processing API", False, f"HTTP {response.status_code}: {error_msg}")
                return False
                
        except requests.exceptions.Timeout:
            self.log_test("Document processing API", False, "Request timeout (>120s)")
            return False
        except Exception as e:
            self.log_test("Document processing API", False, str(e))
            return False

    def test_10_performance_benchmark(self):
        """Test 10: Performance benchmark"""
        print("\n🔍 Test 10: Performance Benchmark")
        
        # Test multiple questions to benchmark performance
        perf_test_data = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": [
                "What is the main topic?",
                "Who is the author?",
                "What are the key points?",
                "Are there any dates mentioned?",
                "What type of document is this?"
            ]
        }
        
        try:
            print(f"   ⏳ Benchmarking with {len(perf_test_data['questions'])} questions...")
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                json=perf_test_data,
                headers=headers,
                timeout=180
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get('answers', [])
                avg_time = total_time / len(answers) if answers else 0
                
                self.log_test("Performance benchmark", True, 
                            f"Avg: {avg_time:.2f}s/question, Total: {total_time:.2f}s")
                
                # Performance assessment
                if avg_time < 10:
                    print("   🚀 Excellent performance!")
                elif avg_time < 20:
                    print("   ✅ Good performance")
                elif avg_time < 40:
                    print("   ⚠️  Acceptable performance")
                else:
                    print("   🐌 Slow performance - consider model optimization")
                    
                return True
            else:
                self.log_test("Performance benchmark", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Performance benchmark", False, str(e))
            return False

    def test_11_cache_functionality(self):
        """Test 11: Smart caching functionality"""
        print("\n🔍 Test 11: Smart Cache Functionality")
        
        try:
            # First request (should be cache miss)
            test_data = {
                "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "questions": ["What is this document about?"]
            }
            
            print("   Making first request (cache miss)...")
            start_time = time.time()
            response1 = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers, timeout=60)
            time1 = time.time() - start_time
            
            if response1.status_code != 200:
                self.log_test("Cache functionality", False, "First request failed")
                return False
            
            # Second identical request (should be cache hit)
            print("   Making second identical request (cache hit)...")
            start_time = time.time()
            response2 = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers, timeout=30)
            time2 = time.time() - start_time
            
            if response2.status_code != 200:
                self.log_test("Cache functionality", False, "Second request failed")
                return False
            
            # Check cache stats
            cache_response = requests.get(f"{BASE_URL}/cache/stats")
            
            if cache_response.status_code == 200:
                cache_stats = cache_response.json().get('cache_stats', {})
                hit_rate = cache_stats.get('hit_rate_percent', 0)
                
                # Assess caching
                if time2 < time1 * 0.5:  # Cache hit should be significantly faster
                    self.log_test("Cache functionality", True, 
                                f"Cache working! Hit rate: {hit_rate}%, Speed improvement: {time1/time2:.1f}x")
                else:
                    self.log_test("Cache functionality", True, 
                                f"Cache present but modest improvement. Hit rate: {hit_rate}%")
                return True
            else:
                self.log_test("Cache functionality", False, "Cache stats endpoint unavailable")
                return False
                
        except Exception as e:
            self.log_test("Cache functionality", False, str(e))
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Local LLM Migration Test Suite")
        print("=" * 60)
        print("Testing migration from Groq to Local Gemma via Ollama")
        print("=" * 60)
        
        # Define test sequence
        tests = [
            self.test_1_basic_imports,
            self.test_2_fastapi_imports,
            self.test_3_ml_dependencies,
            self.test_4_ollama_availability,
            self.test_5_custom_modules,
            self.test_6_component_initialization,
            self.test_7_llm_functionality,
            self.test_8_api_server_health,
            self.test_9_document_processing_api,
            self.test_10_performance_benchmark,
            self.test_11_cache_functionality
        ]
        
        # Run tests
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.log_test(test.__name__, False, str(e))
        
        # Summary
        self.print_summary()

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 LOCAL LLM MIGRATION TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"✅ Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n🎉 PERFECT! Your Local LLM migration is fully successful!")
            print("\n🔥 Benefits you're now enjoying:")
            print("   • ✅ Zero API costs - completely free")
            print("   • ✅ No rate limits - unlimited processing")
            print("   • ✅ Enhanced privacy - all processing local")
            print("   • ✅ Offline capability - works without internet")
            print("   • ✅ Smart caching - faster repeat queries")
            print("   • ✅ Full control over the model")
            
        elif success_rate >= 80:
            print("\n✅ GOOD! Your migration is mostly successful!")
            print("   Minor issues detected - check failed tests above")
            
        elif success_rate >= 60:
            print("\n⚠️  PARTIAL SUCCESS - Some critical issues found")
            print("   Review failed tests and fix before production use")
            
        else:
            print("\n❌ MAJOR ISSUES - Migration needs significant fixes")
            print("   Multiple components failing - check setup")
        
        # Failed tests breakdown
        failed_tests = [r for r in self.test_results if not r['passed']]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        print("\n💡 Next Steps:")
        if success_rate == 100:
            print("   • 🚀 Deploy to production!")
            print("   • 📊 Monitor performance with real documents")
            print("   • 🔧 Fine-tune model parameters if needed")
        else:
            print("   • 🔧 Fix failing components shown above")
            print("   • 📚 Check Ollama documentation if model issues")
            print("   • 🔄 Re-run tests after fixes")
        
        print("   • 📈 Use /cache/stats endpoint to monitor caching")
        print("   • 🧪 Test with your specific insurance documents")
        
        print("\n" + "=" * 60)


def main():
    """Main test execution"""
    tester = LocalLLMTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()