from typing import Dict, List, Any
import logging
import time
import re
import json
from pydantic import BaseModel

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryStructure(BaseModel):
    """Structured representation of a parsed query"""
    intent: str
    entities: Dict[str, Any]
    keywords: List[str]
    domain: str
    complexity: str

class Gemma3Manager:
    """Manages Gemma 3:12B model with optimized settings"""
    
    def __init__(self, model_name: str = "gemma3:12b"):  # Changed to more commonly available model
        self.model_name = model_name
        
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama not available. Install with: pip install ollama")
            
        self._test_model_availability()
        logger.info(f"✓ {model_name} initialized successfully")
    
    def _test_model_availability(self):
        """Test if the model is available"""
        try:
            # Quick test with minimal tokens
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 5}
            )
        except Exception as e:
            logger.error(f"✗ {self.model_name} not available: {e}")
            logger.info(f"Run: ollama pull {self.model_name}")
            raise
    
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using Gemma model"""
        max_tokens = kwargs.get('max_tokens', 2000)
        temperature = kwargs.get('temperature', 0.1)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    # Gemma specific optimizations
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,  # Context window
                    "stop": ["</s>", "[/INST]", "User:", "Assistant:"]
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"{self.model_name} generation failed: {e}")
            raise

class LLMProcessor:
    """LLM processor compatible with your existing code structure"""
    
    def __init__(self, model_name: str = "gemma3:12b", max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.model_manager = Gemma3Manager(model_name=model_name)
        
        logger.info(f"🚀 {model_name} LLM processor initialized")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query parsing for insurance documents"""
        return self._enhanced_fallback_parse_query(query)

    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using local Gemma model"""
        if not retrieved_chunks:
            return self._no_context_answer(question)

        # Prepare context
        context_text = self._prepare_insurance_context(retrieved_chunks)

        try:
            return self._generate_with_gemma(question, context_text, retrieved_chunks)
        except Exception as e:
            logger.warning(f"Gemma generation failed: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)

    def _prepare_insurance_context(self, chunks: List[Dict]) -> str:
        """Prepare context for Gemma model"""
        context_parts = []
        total_length = 0
        max_context_length = 6000  # Reasonable limit

        for i, chunk in enumerate(chunks[:5]):  # Top 5 chunks
            chunk_text = chunk['text']
            
            # Truncate if too long
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
        """Generate with local Gemma model using optimized prompting"""
        
        question_type = self._classify_insurance_question(question)
        
        # System message
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
        
        user_message = self._create_user_message(question, context_text, question_type)
        messages = [system_message, user_message]

        start_time = time.time()
        
        try:
            answer_text = self.model_manager.generate_response(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            cleaned_answer = self._clean_insurance_answer(answer_text)
            
            return {
                "answer": cleaned_answer,
                "reasoning": self._extract_insurance_reasoning(cleaned_answer, question_type),
                "confidence": self._assess_insurance_confidence(retrieved_chunks, cleaned_answer),
                "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(retrieved_chunks[:3])],
                "token_usage": self._estimate_tokens(' '.join([m['content'] for m in messages]) + answer_text),
                "processing_time": processing_time,
                "question_type": question_type,
                "model": self.model_manager.model_name
            }

        except Exception as e:
            logger.error(f"Gemma generation failed: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)

    def _create_user_message(self, question: str, context_text: str, question_type: str) -> Dict[str, str]:
        """Create optimized user message"""
        
        type_instructions = {
            "waiting_period": "Focus on exact waiting periods and their conditions.",
            "coverage": "Identify comprehensive coverage details and limitations.",
            "exclusion": "Find all exclusion clauses and restrictions.",
            "claim_process": "Detail the complete claim procedure.",
            "condition": "Extract all conditions and requirements.",
            "definition": "Provide complete definitions of terms.",
            "time_period": "Analyze all time-related provisions.",
            "amount": "Extract numerical values with precision.",
            "general": "Provide comprehensive analysis."
        }
        
        specific_instruction = type_instructions.get(question_type, type_instructions["general"])
        
        user_content = f"""INSURANCE ANALYSIS REQUEST

QUESTION: {question}

FOCUS: {specific_instruction}

DOCUMENT CONTEXT:
{context_text}

REQUIREMENTS:
- Provide precise quotes for factual claims
- Identify conditional relationships
- State clearly if information is incomplete
- Be specific with numbers and timeframes

Analyze the document sections above and provide a comprehensive answer."""

        return {"role": "user", "content": user_content}

    def _classify_insurance_question(self, question: str) -> str:
        """Classify insurance question type"""
        question_lower = question.lower()
        
        patterns = {
            'waiting_period': ['waiting period', 'wait', 'cooling period'],
            'grace_period': ['grace period', 'grace day'],
            'coverage': ['coverage', 'cover', 'included', 'benefit'],
            'exclusion': ['exclude', 'exclusion', 'not covered', 'limitation'],
            'claim_process': ['claim', 'settlement', 'process', 'submit'],
            'condition': ['condition', 'requirement', 'eligib', 'criteria'],
            'definition': ['define', 'definition', 'mean', 'what is'],
            'time_period': ['period', 'duration', 'time', 'day', 'month'],
            'amount': ['amount', 'limit', 'maximum', 'sum insured'],
            'pre_existing': ['pre-existing', 'PED', 'prior condition'],
            'maternity': ['maternity', 'pregnancy', 'childbirth'],
        }
        
        for category, keywords in patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
                
        return "general"

    def _clean_insurance_answer(self, answer_text: str) -> str:
        """Clean and format answer"""
        prefixes_to_remove = [
            r'^(Answer:|ANSWER:|Response:|Based on the document[s]?:)\s*',
            r'^(According to the document[s]?:)\s*',
            r'^(Analysis:)\s*'
        ]
        
        for prefix in prefixes_to_remove:
            answer_text = re.sub(prefix, '', answer_text, flags=re.IGNORECASE)
        
        # Clean formatting
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        
        # Ensure proper punctuation
        if answer_text and not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
            
        return answer_text

    def _extract_insurance_reasoning(self, answer_text: str, question_type: str) -> str:
        """Extract reasoning from answer"""
        quotes = re.findall(r'"([^"]*)"', answer_text)
        if quotes:
            return f"Based on policy text: {quotes[0][:150]}..."
        
        reasoning_indicators = [
            r'according to[^.]*\.',
            r'as stated[^.]*\.',
            r'the policy[^.]*\.',
        ]
        
        for pattern in reasoning_indicators:
            matches = re.finditer(pattern, answer_text, re.IGNORECASE)
            for match in matches:
                return match.group(0).strip()
        
        return f"Analysis of {question_type} provisions from document sections."

    def _assess_insurance_confidence(self, chunks: List[Dict], answer: str) -> str:
        """Assess confidence level"""
        if not chunks:
            return "low"
            
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # Check for quality indicators
        has_quotes = '"' in answer or "'" in answer
        has_specifics = bool(re.search(r'\d+\s*(day|month|year|%|₹|\$)', answer, re.IGNORECASE))
        has_conditions = bool(re.search(r'(condition|subject to|provided|if|unless)', answer, re.IGNORECASE))
        
        confidence_score = avg_score
        if has_quotes: confidence_score += 0.1
        if has_specifics: confidence_score += 0.1
        if has_conditions: confidence_score += 0.05
        
        if confidence_score > 0.8:
            return "high"
        elif confidence_score > 0.6:
            return "medium"
        else:
            return "low"

    def _enhanced_fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback query parsing"""
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
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        # Time periods
        time_matches = re.findall(r'(\d+)\s*(day|month|year)s?', query_lower)
        if time_matches:
            entities['time_periods'] = [{'value': int(m[0]), 'unit': m[1]} for m in time_matches]
        
        # Keywords
        keywords = re.findall(r'\b\w+\b', query_lower)
        keywords = [k for k in keywords if len(k) > 2]  # Filter short words
        
        # Domain classification
        insurance_keywords = ['policy', 'coverage', 'premium', 'claim', 'benefit', 'waiting', 'exclusion']
        domain = "insurance" if any(kw in query_lower for kw in insurance_keywords) else "general"
        
        # Complexity assessment
        complexity = "simple" if len(query.split()) < 10 else "complex"
        
        return {
            "intent": intent,
            "entities": entities,
            "keywords": keywords[:10],  # Limit keywords
            "domain": domain,
            "complexity": complexity,
            "original_query": query
        }

    def _enhanced_fallback_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Enhanced fallback answer when LLM fails"""
        if not chunks:
            return {
                "answer": "I could not find relevant information in the document to answer your question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0,
                "processing_time": 0.01,
                "model": "fallback"
            }
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        best_chunk = sorted_chunks[0]
        
        # Try to extract answer using pattern matching
        answer = self._extract_answer_with_rules(question, best_chunk['text'])
        
        if not answer:
            # Fallback to chunk summary
            answer = f"Based on the document: {best_chunk['text'][:300]}..."
        
        return {
            "answer": answer,
            "reasoning": f"Answer extracted from document section with {best_chunk.get('relevance_score', 0):.1%} relevance match.",
            "confidence": self._assess_confidence_from_score(best_chunk.get('relevance_score', 0)),
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(sorted_chunks[:3])],
            "token_usage": 0,
            "processing_time": 0.01,
            "model": "fallback"
        }
    
    def _extract_answer_with_rules(self, question: str, text: str) -> str:
        """Extract answers using rule-based pattern matching"""
        question_lower = question.lower()
        text_lower = text.lower()
        
        # Common question patterns
        patterns = {
            'waiting period': r'waiting period[^\d]*(\d+)\s*(month|year|day)s?',
            'grace period': r'grace period[^\d]*(\d+)\s*(day|month)s?',
            'coverage': r'(cover|coverage)[^.]*[.!]',
            'percentage': r'(\d+)%',
            'amount': r'(\d+(?:,\d{3})*)\s*(rupee|dollar|\$|₹)',
        }
        
        # Check which pattern matches the question
        for pattern_name, pattern in patterns.items():
            if pattern_name in question_lower:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Find the sentence containing this match
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if match.group(0).lower() in sentence.lower():
                            return sentence.strip() + "."
        
        # Fallback: find best matching sentence
        question_keywords = re.findall(r'\b\w+\b', question_lower)
        best_sentence = ""
        max_matches = 0
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for keyword in question_keywords if keyword in sentence_lower)
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        return best_sentence + "." if best_sentence else ""
    
    def _assess_confidence_from_score(self, relevance_score: float) -> str:
        """Assess confidence from relevance score"""
        if relevance_score > 0.7:
            return "high"
        elif relevance_score > 0.5:
            return "medium"
        else:
            return "low"
    
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
        return len(text) // 4  # Rough estimation