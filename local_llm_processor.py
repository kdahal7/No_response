from typing import Dict, List, Any
import logging
import time
import re
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Local LLM processor using Ollama and Gemma"""
    
    def __init__(self, model_name: str = "gemma3:12b", max_tokens: int = 2000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama not available. Install with: pip install ollama")
            
        self._test_model_availability()
        logger.info(f"🚀 {model_name} LLM processor initialized")

    def _test_model_availability(self):
        """Test if the model is available"""
        try:
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 5}
            )
        except Exception as e:
            logger.error(f"✗ {self.model_name} not available: {e}")
            logger.info(f"Run: ollama pull {self.model_name}")
            raise

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query and extract metadata"""
        query_lower = query.lower()
        
        # Extract basic information
        intent = "question"
        if any(word in query_lower for word in ['what', 'how', 'when', 'where', 'why']):
            intent = "information_seeking"
        elif any(word in query_lower for word in ['coverage', 'covered', 'include']):
            intent = "coverage_inquiry"
        elif any(word in query_lower for word in ['waiting', 'period', 'time']):
            intent = "time_inquiry"
        
        # Extract entities
        entities = {}
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        time_matches = re.findall(r'(\d+)\s*(day|month|year)s?', query_lower)
        if time_matches:
            entities['time_periods'] = [{'value': int(m[0]), 'unit': m[1]} for m in time_matches]
        
        keywords = re.findall(r'\b\w+\b', query_lower)
        keywords = [k for k in keywords if len(k) > 2]
        
        domain = "insurance" if any(kw in query_lower for kw in ['policy', 'coverage', 'premium', 'claim']) else "general"
        complexity = "simple" if len(query.split()) < 10 else "complex"
        
        return {
            "intent": intent,
            "entities": entities,
            "keywords": keywords[:10],
            "domain": domain,
            "complexity": complexity,
            "original_query": query
        }

    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using local Gemma model"""
        if not retrieved_chunks:
            return self._no_context_answer(question)

        context_text = self._prepare_context(retrieved_chunks)

        try:
            return self._generate_with_gemma(question, context_text, retrieved_chunks)
        except Exception as e:
            logger.warning(f"Gemma generation failed: {str(e)}")
            return self._fallback_answer(question, retrieved_chunks)

    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context for the model"""
        context_parts = []
        total_length = 0
        max_context_length = 6000

        for i, chunk in enumerate(chunks[:5]):
            chunk_text = chunk['text']
            
            if len(chunk_text) > 1200:
                sentences = chunk_text.split('. ')
                truncated = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 1150:
                        break
                    truncated.append(sentence)
                    current_length += len(sentence)
                
                chunk_text = '. '.join(truncated) + '.'
            
            if total_length + len(chunk_text) > max_context_length:
                break
                
            relevance = chunk.get('relevance_score', 0)
            context_parts.append(f"[Document Section {i+1} - Relevance: {relevance:.3f}]\n{chunk_text}")
            total_length += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def _generate_with_gemma(self, question: str, context_text: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer with Gemma model"""
        
        system_message = {
            "role": "system", 
            "content": """You are an expert insurance document analyst.

ANALYSIS STANDARDS:
1. PRECISION: Quote exact text from documents for factual claims
2. ACCURACY: Be precise with numbers, dates, and conditions
3. COMPLETENESS: Address all aspects thoroughly
4. CLARITY: Structure answers logically
5. HONESTY: State clearly if information is missing

FOCUS AREAS:
- Coverage vs exclusions
- Waiting periods and time conditions
- Numerical limits and amounts
- Conditional clauses
- Policy terms and definitions

FORMAT:
- Direct answer first
- Support with document quotes
- List conditions/limitations
- Address multiple scenarios if applicable"""
        }
        
        user_content = f"""INSURANCE ANALYSIS REQUEST

QUESTION: {question}

DOCUMENT CONTEXT:
{context_text}

REQUIREMENTS:
- Provide precise quotes for factual claims
- Identify conditional relationships
- State clearly if information is incomplete
- Be specific with numbers and timeframes

Analyze the document sections above and provide a comprehensive answer."""

        user_message = {"role": "user", "content": user_content}
        messages = [system_message, user_message]

        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,
                    "stop": ["</s>", "[/INST]", "User:", "Assistant:"]
                }
            )
            
            answer_text = response['message']['content']
            processing_time = time.time() - start_time
            
            cleaned_answer = self._clean_answer(answer_text)
            
            return {
                "answer": cleaned_answer,
                "reasoning": f"Analysis based on {len(retrieved_chunks)} document sections",
                "confidence": self._assess_confidence(retrieved_chunks, cleaned_answer),
                "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(retrieved_chunks[:3])],
                "token_usage": self._estimate_tokens(' '.join([m['content'] for m in messages]) + answer_text),
                "processing_time": processing_time,
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"Gemma generation failed: {str(e)}")
            return self._fallback_answer(question, retrieved_chunks)

    def _clean_answer(self, answer_text: str) -> str:
        """Clean and format answer"""
        prefixes_to_remove = [
            r'^(Answer:|ANSWER:|Response:|Based on the document[s]?:)\s*',
            r'^(According to the document[s]?:)\s*',
            r'^(Analysis:)\s*'
        ]
        
        for prefix in prefixes_to_remove:
            answer_text = re.sub(prefix, '', answer_text, flags=re.IGNORECASE)
        
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        
        if answer_text and not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
            
        return answer_text

    def _assess_confidence(self, chunks: List[Dict], answer: str) -> str:
        """Assess confidence level"""
        if not chunks:
            return "low"
            
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        has_quotes = '"' in answer or "'" in answer
        has_specifics = bool(re.search(r'\d+\s*(day|month|year|%|₹|\$)', answer, re.IGNORECASE))
        
        confidence_score = avg_score
        if has_quotes: confidence_score += 0.1
        if has_specifics: confidence_score += 0.1
        
        if confidence_score > 0.8:
            return "high"
        elif confidence_score > 0.6:
            return "medium"
        else:
            return "low"

    def _fallback_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback answer when model fails"""
        if not chunks:
            return self._no_context_answer(question)
        
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        best_chunk = sorted_chunks[0]
        
        answer = f"Based on the document: {best_chunk['text'][:300]}..."
        
        return {
            "answer": answer,
            "reasoning": f"Fallback answer from document section with {best_chunk.get('relevance_score', 0):.1%} relevance",
            "confidence": "low",
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(sorted_chunks[:3])],
            "token_usage": 0,
            "processing_time": 0.01,
            "model": "fallback"
        }
    
    def _no_context_answer(self, question: str) -> Dict[str, Any]:
        """Answer when no context is available"""
        return {
            "answer": "I could not find relevant information in the document to answer this question. Please ensure the document contains the requested information or try rephrasing your question.",
            "reasoning": "No relevant document sections were retrieved for this question.",
            "confidence": "low",
            "supporting_chunks": [],
            "token_usage": 0,
            "processing_time": 0.01,
            "model": "no_context"
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4