import streamlit as st
import pandas as pd
from src.rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

# Initialize RAG pipeline
if 'rag_pipeline' not in st.session_state:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        st.session_state.rag_pipeline = RAGPipeline(groq_api_key)
        # Initialize if not already done
        if not st.session_state.rag_pipeline.is_initialized:
            data_folder = os.path.join(os.path.dirname(__file__), "data")
            st.session_state.rag_pipeline.initialize(data_folder)

st.title("📊 LLM-as-a-Judge Evaluation Dashboard")

# Get analytics
analytics = st.session_state.rag_pipeline.get_evaluation_analytics()

if "total_evaluations" not in analytics:
    st.info("No evaluations have been performed yet. Enable evaluation in the main chat to start collecting data.")
else:
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Evaluations", analytics["total_evaluations"])
    
    with col2:
        avg_score = analytics["overall_metrics"]["average_score"]
        st.metric("Average Score", f"{avg_score}/5.0")
    
    with col3:
        # Performance level
        if avg_score >= 4.0:
            performance = "🟢 Excellent"
        elif avg_score >= 3.0:
            performance = "🟡 Good"
        else:
            performance = "🔴 Needs Improvement"
        st.metric("System Performance", performance)
    
    with col4:
        highest_dim = max(analytics["overall_metrics"].items(), key=lambda x: x[1] if x[0] != "average_score" else 0)
        st.metric("Strongest Dimension", highest_dim[0].replace('_', ' ').title())
    
    st.markdown("---")
    
    # Dimension breakdown
    st.subheader("📋 Performance by Dimension")
    
    # Create DataFrame for better visualization
    dimensions_data = []
    for dim, score in analytics["overall_metrics"].items():
        if dim != "average_score":
            dimensions_data.append({
                "Dimension": dim.replace('_', ' ').title(),
                "Average Score": score,
                "Performance": "🟢" if score >= 4.0 else "🟡" if score >= 3.0 else "🔴"
            })
    
    df = pd.DataFrame(dimensions_data)
    st.dataframe(df, use_container_width=True)
    
    # Bar chart
    st.subheader("📊 Score Distribution")
    chart_data = df.set_index('Dimension')['Average Score']
    st.bar_chart(chart_data)
    
    # Session-wise evaluation
    st.subheader("💬 Session Evaluations")
    
    sessions = st.session_state.rag_pipeline.get_sessions()
    if sessions:
        selected_session = st.selectbox("Select Session to View Evaluations:", sessions)
        
        if selected_session:
            session_evals = st.session_state.rag_pipeline.get_session_evaluation_summary(selected_session)
            
            if session_evals:
                st.write(f"Found {len(session_evals)} evaluations for session {selected_session[:8]}...")
                
                # Create detailed evaluation table
                eval_df_data = []
                for i, eval_data in enumerate(session_evals):
                    eval_df_data.append({
                        "Q#": i+1,
                        "Query": eval_data['query'][:100] + "..." if len(eval_data['query']) > 100 else eval_data['query'],
                        "Overall Score": eval_data['overall_score'],
                        "Factual Accuracy": eval_data['scores'].get('factual_accuracy', 0),
                        "Legal Reasoning": eval_data['scores'].get('legal_reasoning', 0),
                        "Citation Quality": eval_data['scores'].get('citation_quality', 0),
                        "Clarity": eval_data['scores'].get('clarity', 0),
                        "Completeness": eval_data['scores'].get('completeness', 0),
                        "Relevance": eval_data['scores'].get('relevance', 0)
                    })
                
                eval_df = pd.DataFrame(eval_df_data)
                st.dataframe(eval_df, use_container_width=True)
                
            else:
                st.info("No evaluations found for this session")
    else:
        st.info("No active sessions found")
