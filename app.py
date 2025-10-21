import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal RAG Chatbot",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        st.session_state.rag_pipeline = RAGPipeline(groq_api_key)
    else:
        st.error("Please set your GROQ_API_KEY in the .env file")
        st.stop()

if 'current_session' not in st.session_state:
    st.session_state.current_session = None

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar for session management
with st.sidebar:
    st.title("📝 Session Management")
    
    # Initialize system
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing RAG pipeline..."):
                data_folder = os.path.join(os.path.dirname(__file__), "data")
                st.session_state.rag_pipeline.initialize(data_folder)
                st.session_state.initialized = True
                st.success("System initialized!")
                st.rerun()
    else:
        st.success("✅ System Ready")
    
    # Session controls
    if st.session_state.initialized:
        if st.button("🆕 New Chat Session"):
            session_id = st.session_state.rag_pipeline.create_new_session()
            st.session_state.current_session = session_id
            st.success(f"Created new session: {session_id[:8]}...")
            st.rerun()
        
        # Get all sessions and display current session info
        sessions = st.session_state.rag_pipeline.get_sessions()
        
        # Display current session info with token usage
        if st.session_state.current_session:
            # Check if current session still exists in the list
            if st.session_state.current_session not in sessions:
                st.warning("Current session no longer exists. Please select a new session.")
                st.session_state.current_session = None
            else:
                session_info = st.session_state.rag_pipeline.get_session_info(
                    st.session_state.current_session
                )
                
                st.write(f"**Current Session:** {st.session_state.current_session[:8]}...")
                
                # Token usage in sidebar
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tokens", session_info.get("total_context_tokens", 0))
                with col2:
                    st.metric("Messages", session_info.get("total_conversations", 0))
                
                # Progress bar
                progress_value = min(session_info.get("token_usage_percentage", 0) / 100, 1.0)
                st.progress(progress_value)
                
                if session_info.get("approaching_limit", False):
                    st.warning("⚠️ Near token limit!")
                
                if session_info.get("has_summary", False):
                    st.info("📋 Has summary")
                
                if st.button("🗑️ Delete Current Session"):
                    st.session_state.rag_pipeline.delete_session(st.session_state.current_session)
                    st.session_state.current_session = None
                    st.success("Session deleted!")
                    st.rerun()
        
        # List all sessions
        if sessions:
            st.write("**Active Sessions:**")
            st.write(f"Total sessions found: {len(sessions)}")
            
            for session in sessions:
                try:
                    session_info = st.session_state.rag_pipeline.get_session_info(session)
                    
                    # Create a more informative button label
                    summary_indicator = "📋" if session_info.get("has_summary", False) else "📄"
                    total_msgs = session_info.get("total_conversations", 0)
                    last_activity = session_info.get("last_activity")
                    
                    # Format last activity
                    if last_activity:
                        time_str = last_activity.strftime("%m/%d %H:%M")
                    else:
                        time_str = "No activity"
                    
                    button_label = f"{summary_indicator} {session[:8]}... ({total_msgs} msgs) - {time_str}"
                    
                    # Check if this is the current session
                    is_current = session == st.session_state.current_session
                    
                    if is_current:
                        st.info(f"🔸 CURRENT: {button_label}")
                    else:
                        if st.button(button_label, key=f"session_{session}"):
                            st.session_state.current_session = session
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error loading session {session[:8]}: {str(e)}")
        else:
            st.info("No previous sessions found. Create a new session to start chatting!")

# Main chat interface
st.title("⚖️ Legal RAG Chatbot")
st.write("Ask questions about human rights law and legal matters.")

if not st.session_state.initialized:
    st.warning("Please initialize the system first using the sidebar.")
    st.info("Make sure to add your PDF documents to the 'data' folder before initializing.")
elif not st.session_state.current_session:
    st.info("Please create a new chat session to start chatting.")
else:
    # Display conversation history
    session_history = st.session_state.rag_pipeline.conversation_manager.get_session_history(
        st.session_state.current_session
    )
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for user_msg, bot_msg, timestamp in session_history:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(bot_msg)
    
    # Chat input
    if prompt := st.chat_input("Ask your legal question..."):
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
        
        # Generate and display response with debug info
        with st.spinner("Thinking..."):
            # Get conversation context for debugging
            conversation_context = st.session_state.rag_pipeline.conversation_manager.get_conversation_context(
                st.session_state.current_session
            )
            
            # Get retrieved context for debugging
            retrieved_context = st.session_state.rag_pipeline.retrieve_context(prompt)
            
            # Get token usage information
            token_info = st.session_state.rag_pipeline.conversation_manager.get_token_info(
                st.session_state.current_session
            )
            
            # Show debug information in expandable sections
            with st.expander("🔍 Debug Information", expanded=False):
                # Token Usage Section
                st.subheader("📊 Token Usage:")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Context Tokens", 
                        token_info.get("total_context_tokens", 0),
                        delta=f"{token_info.get('token_usage_percentage', 0):.1f}% of limit"
                    )
                
                with col2:
                    st.metric(
                        "Summary Tokens", 
                        token_info.get("summary_tokens", 0)
                    )
                
                with col3:
                    st.metric(
                        "Recent Conv Tokens", 
                        token_info.get("recent_conversation_tokens", 0)
                    )
                
                # Warning if approaching limit
                if token_info.get("approaching_limit", False):
                    st.warning("⚠️ Approaching token limit - summarization may trigger soon!")
                
                # Progress bar for token usage
                progress_value = min(token_info.get("token_usage_percentage", 0) / 100, 1.0)
                st.progress(progress_value, text=f"Token Usage: {token_info.get('total_context_tokens', 0)}/{token_info.get('max_tokens', 2000)}")
                
                st.subheader("Retrieved Context:")
                st.text_area("Context sent to LLM:", retrieved_context, height=150, disabled=True)
                
                st.subheader("Conversation History:")
                st.text_area("Previous exchanges:", conversation_context, height=100, disabled=True)
                
                st.subheader("System Prompt:")
                system_prompt = """You are a helpful legal assistant specializing in human rights law. 
Use the provided context to answer questions accurately and cite relevant information when possible.
If the context doesn't contain relevant information, say so clearly.
Keep your responses professional and helpful."""
                st.text_area("System prompt:", system_prompt, height=100, disabled=True)
            
            response = st.session_state.rag_pipeline.chat(st.session_state.current_session, prompt)
        
        with chat_container:
            with st.chat_message("assistant"):
                st.write(response)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Token usage is tracked in real-time. Summarization triggers at 2000+ tokens to optimize performance.")
