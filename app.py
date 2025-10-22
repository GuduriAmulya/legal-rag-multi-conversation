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
        
        # Generate and display response with optional evaluation
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
                # Add document retrieval debugging
                # st.subheader("🔍 Document Retrieval Debug:")
                
                # # Check if vector store is loaded
                # vector_store_status = st.session_state.rag_pipeline.is_initialized
                # st.write(f"Vector Store Initialized: {vector_store_status}")
                
                # if vector_store_status:
                #     # Check number of documents in vector store
                #     num_docs = len(st.session_state.rag_pipeline.vector_store.documents)
                #     st.write(f"Total Documents in Vector Store: {num_docs}")
                    
                #     # Show sample document chunks
                #     if num_docs > 0:
                #         st.write("**Sample Document Chunks (first 3):**")
                #         for i, doc in enumerate(st.session_state.rag_pipeline.vector_store.documents[:3]):
                #             st.text_area(f"Chunk {i+1}:", doc[:200] + "..." if len(doc) > 200 else doc, height=80, key=f"chunk_{i}")
                    
                #     # Test search with current query
                #     st.write(f"**Search Results for: '{prompt}'**")
                #     search_results = st.session_state.rag_pipeline.vector_store.search(prompt, k=5)
                
                #     if search_results:
                #         for i, (doc, score) in enumerate(search_results):
                #             st.write(f"Result {i+1} (Score: {score:.3f}):")
                #             st.text_area(f"Content {i+1}:", doc[:300] + "..." if len(doc) > 300 else doc, height=80, key=f"result_{i}")
                #     else:
                #         st.error("No search results found!")
                
                # # Token Usage Section
                # st.subheader("📊 Token Usage:")
                # col1, col2, col3 = st.columns(3)
                
                # with col1:
                #     st.metric(
                #         "Total Context Tokens", 
                #         token_info.get("total_context_tokens", 0),
                #         delta=f"{token_info.get('token_usage_percentage', 0):.1f}% of limit"
                #     )
                
                # with col2:
                #     st.metric(
                #         "Summary Tokens", 
                #         token_info.get("summary_tokens", 0)
                #     )
                
                # with col3:
                #     st.metric(
                #         "Recent Conv Tokens", 
                #         token_info.get("recent_conversation_tokens", 0)
                #     )
                
                # # Warning if approaching limit
                # if token_info.get("approaching_limit", False):
                #     st.warning("⚠️ Approaching token limit - summarization may trigger soon!")
                
                # # Progress bar for token usage
                # progress_value = min(token_info.get("token_usage_percentage", 0) / 100, 1.0)
                # st.progress(progress_value, text=f"Token Usage: {token_info.get('total_context_tokens', 0)}/{token_info.get('max_tokens', 2000)}")
                
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
            
            # Use evaluation-enabled chat if enabled
            if st.session_state.get('enable_evaluation', False):
                response, evaluation = st.session_state.rag_pipeline.chat_with_evaluation(
                    st.session_state.current_session, prompt, enable_evaluation=True
                )
            else:
                response = st.session_state.rag_pipeline.chat(st.session_state.current_session, prompt)
                evaluation = None
        
        with chat_container:
            with st.chat_message("assistant"):
                st.write(response)
                
                # Show evaluation results if available - ALWAYS VISIBLE
                if evaluation:
                    st.markdown("---")
                    st.subheader("⚖️ LLM Judge Evaluation Results")
                    
                    # Overall score prominently displayed
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        st.metric("📊 Overall Score", f"{evaluation.overall_score:.1f}/5.0", 
                                help="Average of all dimension scores")
                    with col2:
                        # Determine performance level
                        if evaluation.overall_score >= 4.0:
                            performance = "🟢 Excellent"
                        elif evaluation.overall_score >= 3.0:
                            performance = "🟡 Good"
                        else:
                            performance = "🔴 Needs Improvement"
                        st.metric("📈 Performance", performance)
                    with col3:
                        st.metric("🔍 Evaluated", "✅ Complete")
                    
                    # Detailed scores in a more visible format
                    st.subheader("📋 Detailed Dimension Scores")
                    score_cols = st.columns(3)
                    
                    dimensions = list(evaluation.scores.items())
                    for i, (dimension, score) in enumerate(dimensions):
                        col_idx = i % 3
                        with score_cols[col_idx]:
                            # Color-coded scores
                            if score >= 4:
                                color = "🟢"
                            elif score >= 3:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.metric(
                                f"{color} {dimension.replace('_', ' ').title()}", 
                                f"{score}/5",
                                help=evaluation.explanations.get(dimension, "No explanation")
                            )
                    
                    # Expandable detailed explanations
                    with st.expander("📝 Detailed Judge Reasoning", expanded=False):
                        for dimension, explanation in evaluation.explanations.items():
                            st.write(f"**{dimension.replace('_', ' ').title()}:** {explanation}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Token usage is tracked in real-time. Summarization triggers at 2000+ tokens to optimize performance.")

# Add evaluation dashboard in sidebar
if st.session_state.initialized:
        st.markdown("---")
        st.subheader("📊 Evaluation System")
        
        # Toggle evaluation
        if 'enable_evaluation' not in st.session_state:
            st.session_state.enable_evaluation = False
            
        st.session_state.enable_evaluation = st.checkbox(
            "🔍 Enable LLM-as-a-Judge Evaluation", 
            value=st.session_state.enable_evaluation,
            help="Automatically evaluate each response using AI judge"
        )
        
        # Show evaluation analytics
        if st.button("📈 View Analytics"):
            analytics = st.session_state.rag_pipeline.get_evaluation_analytics()
            if "total_evaluations" in analytics:
                st.success(f"📊 {analytics['total_evaluations']} evaluations completed")
                
                # Show metrics in a nice format
                metrics = analytics["overall_metrics"]
                st.write("**Average Scores:**")
                for metric, value in metrics.items():
                    if metric != "average_score":
                        st.write(f"• {metric.replace('_', ' ').title()}: {value}/5.0")
                
            else:
                st.info(analytics["message"])
        
        # Show current session evaluations if available
        if st.session_state.current_session:
            if st.button("📋 Session Evaluations"):
                session_evals = st.session_state.rag_pipeline.get_session_evaluation_summary(
                    st.session_state.current_session
                )
                
                if session_evals:
                    st.write(f"**Evaluations for Current Session ({len(session_evals)}):**")
                    for i, eval_data in enumerate(session_evals):
                        with st.expander(f"Q{i+1}: {eval_data['query'][:50]}... (Score: {eval_data['overall_score']:.1f})"):
                            st.write(f"**Query:** {eval_data['query']}")
                            st.write(f"**Overall Score:** {eval_data['overall_score']:.1f}/5.0")
                            
                            # Show dimension scores
                            cols = st.columns(3)
                            scores = eval_data['scores']
                            score_items = list(scores.items())
                            
                            for j, (dim, score) in enumerate(score_items):
                                col_idx = j % 3
                                with cols[col_idx]:
                                    st.metric(dim.replace('_', ' ').title(), f"{score}/5")
                else:
                    st.info("No evaluations for this session yet")
