import uuid
import sqlite3
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class ConversationManager:
    def __init__(self, max_history: int = 3, db_path: str = None, max_tokens: int = 2000):
        self.max_history = max_history
        self.max_tokens = max_tokens  # Token limit for conversation context
        
        # Set up database path - create in legal_chatbot folder
        if db_path is None:
            current_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to legal_chatbot folder
            self.db_path = os.path.join(current_dir, "conversations.db")
        else:
            self.db_path = db_path
            
        # Initialize database
        self._init_database()
        
        # Keep in-memory cache for performance (will sync with DB)
        self.sessions: Dict[str, List[Tuple[str, str, datetime]]] = {}
        self._load_all_sessions()
        
        # Add summaries storage
        self.session_summaries: Dict[str, str] = {}
        self._load_all_summaries()
    
    def _init_database(self):
        """Initialize SQLite database with conversations table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)
            ''')
            
            # Add summaries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def _load_all_sessions(self):
        """Load all sessions from database into memory cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First, get all unique session IDs
                cursor.execute('SELECT DISTINCT session_id FROM conversations ORDER BY session_id')
                session_ids = [row[0] for row in cursor.fetchall()]
                
                print(f"DEBUG: Found {len(session_ids)} unique sessions in database")
                
                # Load conversations for each session
                for session_id in session_ids:
                    cursor.execute('''
                        SELECT user_message, bot_response, timestamp 
                        FROM conversations 
                        WHERE session_id = ?
                        ORDER BY timestamp
                    ''', (session_id,))
                    
                    rows = cursor.fetchall()
                    self.sessions[session_id] = []
                    
                    for user_msg, bot_msg, timestamp_str in rows:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        self.sessions[session_id].append((user_msg, bot_msg, timestamp))
                    
                    # Apply max_history limit to memory cache (but keep all in DB)
                    if len(self.sessions[session_id]) > self.max_history:
                        self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
                
                print(f"DEBUG: Loaded {len(self.sessions)} sessions into memory")
                
        except Exception as e:
            print(f"Error loading sessions: {e}")
            self.sessions = {}

    def _load_all_summaries(self):
        """Load all conversation summaries from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT session_id, summary FROM conversation_summaries')
            rows = cursor.fetchall()
            
            for session_id, summary in rows:
                self.session_summaries[session_id] = summary
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return len(text) // 4
    
    def _create_summary(self, conversations: List[Tuple[str, str, datetime]], groq_client) -> str:
        """Create a summary of older conversations using Groq API."""
        if not conversations:
            return ""
        
        # Format conversations for summarization
        conversation_text = ""
        for user_msg, bot_msg, timestamp in conversations:
            conversation_text += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        summary_prompt = f"""Please create a concise summary of this legal conversation that preserves key context for future reference:

{conversation_text}

Focus on:
- Main legal topics discussed
- Key facts or cases mentioned  
- User's primary concerns or questions
- Important conclusions reached

Keep the summary under 200 words while preserving essential context."""

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a legal assistant that creates concise conversation summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error creating summary: {e}")
            # Fallback: create simple summary
            topics = []
            for user_msg, bot_msg, _ in conversations:
                if len(user_msg) > 20:  # Only include substantial questions
                    topics.append(user_msg[:100] + "...")
            
            return f"Previous discussion topics: {'; '.join(topics[:3])}"
    
    def _save_summary(self, session_id: str, summary: str):
        """Save conversation summary to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO conversation_summaries (session_id, summary, updated_at)
                VALUES (?, ?, ?)
            ''', (session_id, summary, datetime.now().isoformat()))
            conn.commit()
        
        self.session_summaries[session_id] = summary

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        
        # No need to insert into DB yet - will happen when first exchange is added
        return session_id
    
    def add_exchange(self, session_id: str, user_message: str, bot_response: str):
        """Add a user-bot exchange to the session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        timestamp = datetime.now()
        
        # Add to memory cache
        self.sessions[session_id].append((user_message, bot_response, timestamp))
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (session_id, user_message, bot_response, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_message, bot_response, timestamp.isoformat()))
            conn.commit()
        
        # Keep only the last max_history exchanges in memory
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_conversation_context(self, session_id: str, groq_client=None) -> str:
        """Get optimized conversation context with summarization."""
        if session_id not in self.sessions:
            return ""
        
        recent_conversations = self.sessions[session_id]
        context_parts = []
        
        # Add existing summary if available
        if session_id in self.session_summaries:
            context_parts.append(f"Previous conversation summary:\n{self.session_summaries[session_id]}\n")
        
        # Add recent conversations
        recent_context = []
        for user_msg, bot_msg, _ in recent_conversations:
            recent_context.append(f"User: {user_msg}")
            recent_context.append(f"Assistant: {bot_msg}")
        
        if recent_context:
            context_parts.append("Recent conversation:\n" + "\n".join(recent_context))
        
        full_context = "\n\n".join(context_parts)
        current_tokens = self._estimate_tokens(full_context)
        
        # Log token usage for debugging
        print(f"DEBUG: Current context tokens: {current_tokens}/{self.max_tokens}")
        print(f"DEBUG: Context length (chars): {len(full_context)}")
        print(f"DEBUG: Number of recent conversations: {len(recent_conversations)}")
        print(f"DEBUG: Has summary: {session_id in self.session_summaries}")
        
        # Check token limit and trigger summarization if needed
        if current_tokens > self.max_tokens and groq_client and len(recent_conversations) > 2:
            print(f"⚠️ TRIGGERING SUMMARIZATION: Context too long ({current_tokens} tokens)")
            
            # Get all conversations for this session from database
            all_conversations = self._get_all_session_conversations(session_id)
            print(f"DEBUG: Total conversations in DB: {len(all_conversations)}")
            
            if len(all_conversations) > self.max_history:
                # Conversations to summarize (older ones)
                conversations_to_summarize = all_conversations[:-2]  # All except last 2
                print(f"DEBUG: Summarizing {len(conversations_to_summarize)} older conversations")
                
                # Create summary
                new_summary = self._create_summary(conversations_to_summarize, groq_client)
                print(f"DEBUG: Created summary: {new_summary[:100]}...")
                
                # Save summary
                self._save_summary(session_id, new_summary)
                
                # Rebuild context with new summary
                context_parts = [f"Previous conversation summary:\n{new_summary}\n"]
                
                # Keep only last 2 conversations in recent context
                last_two = recent_conversations[-2:]
                recent_context = []
                for user_msg, bot_msg, _ in last_two:
                    recent_context.append(f"User: {user_msg}")
                    recent_context.append(f"Assistant: {bot_msg}")
                
                if recent_context:
                    context_parts.append("Recent conversation:\n" + "\n".join(recent_context))
                
                full_context = "\n\n".join(context_parts)
                new_tokens = self._estimate_tokens(full_context)
                print(f"✅ AFTER SUMMARIZATION: New token count: {new_tokens}")
        
        return full_context
    
    def _get_all_session_conversations(self, session_id: str) -> List[Tuple[str, str, datetime]]:
        """Get all conversations for a session from database (not limited by max_history)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_message, bot_response, timestamp 
                FROM conversations 
                WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,))
            
            rows = cursor.fetchall()
            conversations = []
            
            for user_msg, bot_msg, timestamp_str in rows:
                timestamp = datetime.fromisoformat(timestamp_str)
                conversations.append((user_msg, bot_msg, timestamp))
            
            return conversations

    def get_session_history(self, session_id: str) -> List[Tuple[str, str, datetime]]:
        """Get the full history for a session."""
        return self.sessions.get(session_id, [])
    
    def delete_session(self, session_id: str):
        """Delete a conversation session."""
        # Remove from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Also remove summary
        if session_id in self.session_summaries:
            del self.session_summaries[session_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM conversation_summaries WHERE session_id = ?', (session_id,))
            conn.commit()
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    def get_session_info(self, session_id: str) -> dict:
        """Get session information including message count and summary status."""
        if session_id not in self.sessions:
            return {}
        
        history = self.sessions[session_id]
        has_summary = session_id in self.session_summaries
        
        return {
            "message_count": len(history),
            "last_activity": history[-1][2] if history else None,
            "session_id_short": session_id[:8],
            "has_summary": has_summary,
            "total_conversations": len(self._get_all_session_conversations(session_id))
        }
    
    def get_token_info(self, session_id: str) -> dict:
        """Get detailed token usage information for a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        # Get current context
        context = self.get_conversation_context(session_id)
        current_tokens = self._estimate_tokens(context)
        
        # Get summary info
        summary_tokens = 0
        if session_id in self.session_summaries:
            summary_tokens = self._estimate_tokens(self.session_summaries[session_id])
        
        # Get recent conversation tokens
        recent_conversations = self.sessions[session_id]
        recent_context = []
        for user_msg, bot_msg, _ in recent_conversations:
            recent_context.append(f"User: {user_msg}")
            recent_context.append(f"Assistant: {bot_msg}")
        
        recent_tokens = self._estimate_tokens("\n".join(recent_context))
        
        return {
            "total_context_tokens": current_tokens,
            "summary_tokens": summary_tokens,
            "recent_conversation_tokens": recent_tokens,
            "max_tokens": self.max_tokens,
            "token_usage_percentage": (current_tokens / self.max_tokens) * 100,
            "approaching_limit": current_tokens > (self.max_tokens * 0.8),
            "context_length_chars": len(context)
        }
