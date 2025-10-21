import os
from groq import Groq
from typing import List, Tuple
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .conversation_manager import ConversationManager

class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.conversation_manager = ConversationManager()
        self.is_initialized = False
    
    def initialize(self, data_folder: str, force_rebuild: bool = False):
        """Initialize the RAG pipeline with documents."""
        # Use relative path from the legal_chatbot directory
        vector_store_path = os.path.join(os.path.dirname(__file__), "..", "vector_store", "legal_docs")
        
        # Try to load existing vector store
        if not force_rebuild and self.vector_store.load(vector_store_path):
            print("Loaded existing vector store.")
            self.is_initialized = True
            return
        
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
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for the query."""
        if not self.is_initialized:
            return ""
        
        results = self.vector_store.search(query, k)
        context_parts = [doc for doc, score in results if score > 0.3]  # Threshold for relevance
        
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
