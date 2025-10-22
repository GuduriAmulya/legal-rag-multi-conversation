import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import sqlite3
import os

@dataclass
class LegalEvaluationResult:
    query: str
    response: str
    retrieval_context: str
    scores: Dict[str, int]
    explanations: Dict[str, str]
    overall_score: float
    timestamp: datetime
    session_id: str

class LegalLLMJudge:
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.evaluation_dimensions = {
            "factual_accuracy": "How factually correct are the legal statements?",
            "legal_reasoning": "Quality of legal analysis and logical flow?", 
            "citation_quality": "Proper use and attribution of legal sources?",
            "clarity": "How clear and understandable is the response?",
            "completeness": "Does it address all aspects of the query?",
            "relevance": "How well does it answer the specific question?"
        }
    
    def evaluate_response(self, query: str, response: str, context: str) -> Dict:
        """Evaluate a legal response using Chain-of-Thought reasoning."""
        
        evaluation_prompt = f"""You are an expert legal evaluator. Analyze this legal Q&A interaction using chain-of-thought reasoning.

QUERY: {query}

RETRIEVED CONTEXT: 
{context}

AI RESPONSE:
{response}

Evaluate the AI response on these 6 dimensions (1-5 scale):

1. **FACTUAL ACCURACY** (1=Incorrect, 5=Completely accurate)
2. **LEGAL REASONING** (1=Poor logic, 5=Excellent legal analysis)  
3. **CITATION QUALITY** (1=No citations, 5=Perfect attribution)
4. **CLARITY** (1=Confusing, 5=Very clear)
5. **COMPLETENESS** (1=Incomplete, 5=Comprehensive)
6. **RELEVANCE** (1=Off-topic, 5=Directly answers question)

For EACH dimension, provide:
- Score (1-5)
- Chain-of-thought explanation (2-3 sentences explaining WHY this score)
- Specific examples from the response

Format your response as JSON:
{{
    "factual_accuracy": {{
        "score": X,
        "reasoning": "Step-by-step explanation...",
        "evidence": "Specific quote or example from response"
    }},
    "legal_reasoning": {{
        "score": X, 
        "reasoning": "...",
        "evidence": "..."
    }},
    [continue for all 6 dimensions],
    "overall_assessment": {{
        "average_score": X.X,
        "strengths": ["strength 1", "strength 2"],
        "weaknesses": ["weakness 1", "weakness 2"],
        "improvement_suggestions": ["suggestion 1", "suggestion 2"]
    }}
}}
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert legal evaluator providing detailed chain-of-thought analysis. Always respond in valid JSON format."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse JSON response
            evaluation_result = json.loads(response.choices[0].message.content)
            return evaluation_result
            
        except Exception as e:
            print(f"Error in LLM Judge evaluation: {e}")
            return self._fallback_evaluation()
    
    def _fallback_evaluation(self) -> Dict:
        """Fallback evaluation if LLM fails."""
        return {
            "factual_accuracy": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "legal_reasoning": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "citation_quality": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "clarity": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "completeness": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "relevance": {"score": 3, "reasoning": "Unable to evaluate", "evidence": "N/A"},
            "overall_assessment": {
                "average_score": 3.0,
                "strengths": ["Evaluation failed"],
                "weaknesses": ["Could not assess"],
                "improvement_suggestions": ["Manual review needed"]
            }
        }

class LegalEvaluationManager:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.judge = LegalLLMJudge(rag_pipeline.groq_client)
        self.db_path = os.path.join(os.path.dirname(__file__), "..", "evaluations.db")
        self._init_evaluation_db()
    
    def _init_evaluation_db(self):
        """Initialize evaluation database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT NOT NULL,
                    factual_accuracy INTEGER,
                    legal_reasoning INTEGER,
                    citation_quality INTEGER,
                    clarity INTEGER,
                    completeness INTEGER,
                    relevance INTEGER,
                    overall_score REAL,
                    evaluation_data TEXT,
                    timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def evaluate_conversation_turn(self, session_id: str, query: str, response: str) -> LegalEvaluationResult:
        """Evaluate a single conversation turn."""
        
        # Get the context that was used for this response
        context = self.rag_pipeline.retrieve_context(query)
        
        # Get LLM Judge evaluation
        evaluation = self.judge.evaluate_response(query, response, context)
        
        # Extract scores
        scores = {}
        explanations = {}
        
        for dimension in self.judge.evaluation_dimensions.keys():
            if dimension in evaluation:
                scores[dimension] = evaluation[dimension].get("score", 3)
                explanations[dimension] = evaluation[dimension].get("reasoning", "No explanation")
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 3.0
        
        # Create result object
        result = LegalEvaluationResult(
            query=query,
            response=response,
            retrieval_context=context,
            scores=scores,
            explanations=explanations,
            overall_score=overall_score,
            timestamp=datetime.now(),
            session_id=session_id
        )
        
        # Save to database
        self._save_evaluation(result, evaluation)
        
        return result
    
    def _save_evaluation(self, result: LegalEvaluationResult, full_evaluation: Dict):
        """Save evaluation to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations (
                    session_id, query, response, context,
                    factual_accuracy, legal_reasoning, citation_quality,
                    clarity, completeness, relevance, overall_score,
                    evaluation_data, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.session_id,
                result.query,
                result.response,
                result.retrieval_context,
                result.scores.get("factual_accuracy", 3),
                result.scores.get("legal_reasoning", 3),
                result.scores.get("citation_quality", 3),
                result.scores.get("clarity", 3),
                result.scores.get("completeness", 3),
                result.scores.get("relevance", 3),
                result.overall_score,
                json.dumps(full_evaluation),
                result.timestamp.isoformat()
            ))
            conn.commit()
    
    def get_evaluation_analytics(self) -> Dict:
        """Get analytics on evaluation results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(overall_score) as avg_overall_score,
                    AVG(factual_accuracy) as avg_factual_accuracy,
                    AVG(legal_reasoning) as avg_legal_reasoning,
                    AVG(citation_quality) as avg_citation_quality,
                    AVG(clarity) as avg_clarity,
                    AVG(completeness) as avg_completeness,
                    AVG(relevance) as avg_relevance
                FROM evaluations
            ''')
            
            stats = cursor.fetchone()
            
            if stats[0] == 0:  # No evaluations yet
                return {"message": "No evaluations performed yet"}
            
            return {
                "total_evaluations": stats[0],
                "overall_metrics": {
                    "average_score": round(stats[1], 2),
                    "factual_accuracy": round(stats[2], 2),
                    "legal_reasoning": round(stats[3], 2),
                    "citation_quality": round(stats[4], 2),
                    "clarity": round(stats[5], 2),
                    "completeness": round(stats[6], 2),
                    "relevance": round(stats[7], 2)
                }
            }
    
    def get_session_evaluations(self, session_id: str) -> List[Dict]:
        """Get all evaluations for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT query, response, overall_score, factual_accuracy,
                       legal_reasoning, citation_quality, clarity,
                       completeness, relevance, timestamp
                FROM evaluations 
                WHERE session_id = ?
                ORDER BY timestamp DESC
            ''', (session_id,))
            
            rows = cursor.fetchall()
            evaluations = []
            
            for row in rows:
                evaluations.append({
                    "query": row[0],
                    "response": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    "overall_score": row[2],
                    "scores": {
                        "factual_accuracy": row[3],
                        "legal_reasoning": row[4],
                        "citation_quality": row[5],
                        "clarity": row[6],
                        "completeness": row[7],
                        "relevance": row[8]
                    },
                    "timestamp": row[9]
                })
            
            return evaluations
