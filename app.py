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
        
        # Display current session
        if st.session_state.current_session:
            st.write(f"**Current Session:** {st.session_state.current_session[:8]}...")
            
            if st.button("🗑️ Delete Current Session"):
                st.session_state.rag_pipeline.delete_session(st.session_state.current_session)
                st.session_state.current_session = None
                st.success("Session deleted!")
                st.rerun()
        
        # List all sessions
        sessions = st.session_state.rag_pipeline.get_sessions()
        if sessions:
            st.write("**Active Sessions:**")
            for session in sessions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📄 {session[:8]}...", key=f"session_{session}"):
                        st.session_state.current_session = session
                        st.rerun()

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
            
            # Show debug information in expandable sections
            with st.expander("🔍 Debug Information", expanded=False):
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
st.markdown("💡 **Tip:** This chatbot maintains conversation context for up to 3 previous exchanges per session.")
