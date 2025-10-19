import uuid
from typing import Dict, List, Tuple
from datetime import datetime

class ConversationManager:
    def __init__(self, max_history: int = 3):
        self.sessions: Dict[str, List[Tuple[str, str, datetime]]] = {}
        self.max_history = max_history
    
    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id
    
    def add_exchange(self, session_id: str, user_message: str, bot_response: str):
        """Add a user-bot exchange to the session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        timestamp = datetime.now()
        self.sessions[session_id].append((user_message, bot_response, timestamp))
        
        # Keep only the last max_history exchanges
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get the conversation context for a session."""
        if session_id not in self.sessions:
            return ""
        
        context_parts = []
        for user_msg, bot_msg, _ in self.sessions[session_id]:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_msg}")
        
        return "\n".join(context_parts)
    
    def get_session_history(self, session_id: str) -> List[Tuple[str, str, datetime]]:
        """Get the full history for a session."""
        return self.sessions.get(session_id, [])
    
    def delete_session(self, session_id: str):
        """Delete a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
