import os
from groq import Groq
from typing import List, Tuple, Dict
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .conversation_manager import ConversationManager

# Add this import
try:
    from .legal_evaluator import LegalEvaluationManager
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("Legal evaluator not available - create legal_evaluator.py first")

class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.conversation_manager = ConversationManager()
        self.evaluator = None  # Initialize after system is ready
        self.is_initialized = False
    
    def initialize(self, data_folder: str, force_rebuild: bool = False):
        """Initialize the RAG pipeline with documents."""
        # Use relative path from the legal_chatbot directory
        vector_store_path = os.path.join(os.path.dirname(__file__), "..", "vector_store", "legal_docs")
        
        # Try to load existing vector store
        if not force_rebuild and self.vector_store.load(vector_store_path):
            print("Loaded existing vector store.")
            self.is_initialized = True
            # Don't return here - continue to initialize evaluator
        else:
            # Process documents and build vector store
            print("Processing documents...")
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                print(f"Created data folder: {data_folder}")
                print("Please add your PDF documents to the data folder.")
                return
            
            chunks = self.document_processor.process_documents(data_folder)
            
            if not chunks:
                print("No documents found to process.")
                return
            
            print(f"Processed {len(chunks)} text chunks.")
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save vector store - create directory relative to legal_chatbot folder
            vector_store_dir = os.path.join(os.path.dirname(__file__), "..", "vector_store")
            os.makedirs(vector_store_dir, exist_ok=True)
            self.vector_store.save(vector_store_path)
            
            self.is_initialized = True
            print("RAG pipeline initialized successfully.")
        
        # Initialize evaluator after system is ready (moved outside the if/else)
        if EVALUATOR_AVAILABLE:
            try:
                self.evaluator = LegalEvaluationManager(self)
                print("✅ Legal Evaluator initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize evaluator: {e}")
                self.evaluator = None
        else:
            print("⚠️ Legal Evaluator not available")

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for the query."""
        if not self.is_initialized:
            print("DEBUG: Vector store not initialized!")
            return ""
        
        print(f"DEBUG: Searching for query: '{query}'")
        print(f"DEBUG: Number of documents in vector store: {len(self.vector_store.documents)}")
        
        results = self.vector_store.search(query, k)
        print(f"DEBUG: Search returned {len(results)} results")
        
        for i, (doc, score) in enumerate(results):
            print(f"DEBUG: Result {i+1} - Score: {score:.3f}, Content: {doc[:100]}...")
        
        context_parts = [doc for doc, score in results if score > 0.3]  # Threshold for relevance
        print(f"DEBUG: After filtering (score > 0.3): {len(context_parts)} documents")
        
        if not context_parts:
            print("DEBUG: No documents passed the relevance threshold!")
            # Lower the threshold temporarily for debugging
            context_parts = [doc for doc, score in results if score > 0.1]
            print(f"DEBUG: With lower threshold (0.1): {len(context_parts)} documents")
        
        final_context = "\n\n".join(context_parts)
        print(f"DEBUG: Final context length: {len(final_context)} characters")
        
        return final_context

    def specialized_retrieval(self, query: str, domain: str) -> str:
        """Perform domain-specific retrieval with enhanced search terms."""
        
        # Domain-specific search enhancement
        search_enhancements = {
            'constitutional': 'constitution article fundamental rights directive principles',
            'criminal': 'IPC criminal law penal code police investigation',
            'civil': 'civil law contract tort damages procedure',
            'family': 'family law marriage divorce personal law',
            'corporate': 'company law corporate governance director liability',
            'property': 'property law ownership transfer registration'
        }
        
        # Enhanced query for better retrieval
        enhanced_query = f"{query} {search_enhancements.get(domain, '')}"
        print(f"DEBUG: Enhanced query for {domain}: '{enhanced_query}'")
        
        # Get more results for complex queries
        k = 5 if domain != 'general' else 3
        
        if not self.is_initialized:
            return ""
        
        results = self.vector_store.search(enhanced_query, k)
        print(f"DEBUG: Specialized search returned {len(results)} results")
        
        # Lower threshold for legal docs to capture more relevant content
        context_parts = [doc for doc, score in results if score > 0.25]  
        print(f"DEBUG: After filtering (score > 0.25): {len(context_parts)} documents")
        
        if not context_parts:
            print("DEBUG: No documents passed specialized threshold, using any results...")
            context_parts = [doc for doc, score in results[:2]]  # Take top 2 regardless of score
        
        return "\n\n".join(context_parts)

    def generate_response(self, query: str, context: str, conversation_context: str = "") -> str:
        """Generate response using Groq."""
        system_prompt = """You are a helpful legal assistant specializing in human rights law. 
        Use the provided context to answer questions accurately and cite relevant information when possible.
        If the context doesn't contain relevant information, say so clearly.
        Keep your responses professional and helpful."""
        
        user_prompt = f"""
        Previous conversation:
        {conversation_context}
        
        Relevant context:
        {context}
        
        Current question: {query}
        
        Please provide a helpful response based on the context and conversation history.
        """
        
        # Print debug information
        print("\n" + "="*80)
        print("DEBUG: SYSTEM PROMPT")
        print("="*80)
        print(system_prompt)
        print("\n" + "="*80)
        print("DEBUG: USER PROMPT (Full Context)")
        print("="*80)
        print(user_prompt)
        print("="*80)
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            generated_response = response.choices[0].message.content
            
            # Print the response for debugging
            print("\n" + "="*80)
            print("DEBUG: GENERATED RESPONSE")
            print("="*80)
            print(generated_response)
            print("="*80 + "\n")
            
            return generated_response
        
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error generating a response: {str(e)}"
            print(f"\nDEBUG: ERROR - {error_msg}\n")
            return error_msg
    
    def chat(self, session_id: str, query: str) -> str:
        """Process a chat message with conversation context."""
        if not self.is_initialized:
            return "The system is not initialized. Please check if documents are loaded."
        
        # STEP 1: Get conversation history with summarization support
        conversation_context = self.conversation_manager.get_conversation_context(
            session_id, 
            groq_client=self.groq_client  # Pass Groq client for summarization
        )
        
        # STEP 2: Create embedding for current query & search vector DB
        context = self.retrieve_context(query)  # This creates embeddings and searches
        
        # STEP 3: Generate response using LLM with both contexts
        response = self.generate_response(query, context, conversation_context)
        
        # STEP 4: Store the current exchange in conversation history
        self.conversation_manager.add_exchange(session_id, query, response)
        
        return response
    
    def chat_with_evaluation(self, session_id: str, query: str, enable_evaluation: bool = True) -> Tuple[str, Dict]:
        """Enhanced chat with optional evaluation."""
        response = self.chat(session_id, query)
        
        evaluation_result = None
        if enable_evaluation and self.evaluator:
            print("🔍 Evaluating response with LLM Judge...")
            try:
                evaluation_result = self.evaluator.evaluate_conversation_turn(session_id, query, response)
                print(f"✅ Evaluation complete - Overall Score: {evaluation_result.overall_score:.1f}/5.0")
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                evaluation_result = None
        
        return response, evaluation_result
    
    def get_evaluation_analytics(self) -> Dict:
        """Get system-wide evaluation analytics."""
        if self.evaluator:
            try:
                return self.evaluator.get_evaluation_analytics()
            except Exception as e:
                return {"message": f"Error getting analytics: {e}"}
        return {"message": "Evaluator not initialized - create legal_evaluator.py first"}
    
    def get_session_evaluation_summary(self, session_id: str) -> Dict:
        """Get evaluation summary for a session."""
        if self.evaluator:
            try:
                return self.evaluator.get_session_evaluations(session_id)
            except Exception as e:
                return {"message": f"Error getting session evaluations: {e}"}
        return {"message": "Evaluator not initialized"}

    def create_new_session(self) -> str:
        """Create a new conversation session."""
        return self.conversation_manager.create_session()
    
    def get_sessions(self) -> List[str]:
        """Get list of active sessions."""
        return self.conversation_manager.list_sessions()
    
    def get_session_info(self, session_id: str) -> dict:
        """Get session information including token usage."""
        session_info = self.conversation_manager.get_session_info(session_id)
        token_info = self.conversation_manager.get_token_info(session_id)
        
        # Combine both info dictionaries
        combined_info = {**session_info, **token_info}
        return combined_info
    
    def delete_session(self, session_id: str):
        """Delete a conversation session."""
        self.conversation_manager.delete_session(session_id)
