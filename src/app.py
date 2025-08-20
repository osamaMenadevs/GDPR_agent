import os
# Set API key FIRST before any other imports
os.environ['TEAM_API_KEY'] = "c484547247c5e2fd0a8a8b38cf75f6721d4b376fef5f596b8941cdd9af7795a9"
os.environ['AIXPLAIN_API_KEY'] = "c484547247c5e2fd0a8a8b38cf75f6721d4b376fef5f596b8941cdd9af7795a9"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import Dict, List, Any
import time

# Import our custom modules
from gdpr_agent import get_gdpr_agent, GDPRComplianceAgent
from data_processor import GDPRDataProcessor
from config import STREAMLIT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f0f8ff;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
        background-color: #fff5f5;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #51cf66;
        background-color: #f3fff3;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False

@st.cache_resource
def initialize_agent():
    """Initialize the GDPR agent (cached for performance)"""
    try:
        with st.spinner("🚀 Initializing GDPR Compliance Agent..."):
            agent = get_gdpr_agent()
            st.session_state.agent_initialized = True
            return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        # Try to load the CSV files
        gdpr_text_df = pd.read_csv("gdpr_text.csv")
        violations_df = pd.read_csv("gdpr_violations.csv")
        return gdpr_text_df, violations_df
    except Exception as e:
        st.warning(f"Could not load data files: {str(e)}")
        return None, None

def display_violation_analytics(violations_df):
    """Display analytics dashboard for GDPR violations"""
    st.markdown('<h2 class="sub-header">📊 GDPR Violations Analytics</h2>', unsafe_allow_html=True)
    
    if violations_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = len(violations_df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Cases</h3>
                <h2 style="color: #1f77b4;">{total_cases}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_fines = violations_df['price'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Fines</h3>
                <h2 style="color: #ff6b6b;">€{total_fines:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_fine = violations_df['price'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Fine</h3>
                <h2 style="color: #51cf66;">€{avg_fine:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            countries = violations_df['name'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Countries</h3>
                <h2 style="color: #ffd43b;">{countries}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Fines by country
            country_fines = violations_df.groupby('name')['price'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=country_fines.index,
                y=country_fines.values,
                title="Top 10 Countries by Total Fines",
                labels={'x': 'Country', 'y': 'Total Fines (€)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violation types
            violation_types = violations_df['type'].value_counts().head(8)
            fig = px.pie(
                values=violation_types.values,
                names=violation_types.index,
                title="Violation Types Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline of violations
        if 'date' in violations_df.columns:
            violations_df['date'] = pd.to_datetime(violations_df['date'], errors='coerce')
            monthly_violations = violations_df.groupby(violations_df['date'].dt.to_period('M')).size()
            
            fig = px.line(
                x=monthly_violations.index.astype(str),
                y=monthly_violations.values,
                title="GDPR Violations Over Time",
                labels={'x': 'Month', 'y': 'Number of Violations'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔒 GDPR Compliance Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Your AI-powered assistant for GDPR compliance analysis, privacy policy evaluation, and regulatory guidance.</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ System Status")
        
        # Initialize agent
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize Agent", type="primary"):
                agent = initialize_agent()
                if agent:
                    st.success("✅ Agent initialized successfully!")
                    st.session_state.agent_initialized = True
        else:
            st.success("✅ Agent Ready")
        
        # Upload embeddings section
        st.markdown("### 📤 Upload Embeddings")
        st.markdown("""
        <div class="info-box">
            <small>Upload your pre-computed embeddings to enhance the RAG system performance.</small>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_embeddings = st.file_uploader(
            "Upload Embeddings File",
            type=['npy', 'pkl'],
            help="Upload numpy array (.npy) or pickle file (.pkl) containing embeddings"
        )
        
        if uploaded_embeddings is not None:
            if st.button("Load Embeddings"):
                try:
                    # Save uploaded file temporarily and load
                    with open(f"temp_{uploaded_embeddings.name}", "wb") as f:
                        f.write(uploaded_embeddings.getbuffer())
                    
                    st.success(f"✅ Embeddings uploaded: {uploaded_embeddings.name}")
                    st.session_state.embeddings_loaded = True
                    
                    # Clean up temp file
                    os.remove(f"temp_{uploaded_embeddings.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading embeddings: {str(e)}")
        
        # System information
        st.markdown("### ℹ️ System Info")
        if st.session_state.agent_initialized:
            try:
                agent = get_gdpr_agent()
                agent_info = agent.get_agent_info()
                st.json(agent_info)
            except:
                st.warning("Agent info unavailable")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Assistant", "🌐 Website Analysis", "📊 Analytics", "📚 Knowledge Base"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">💬 GDPR Compliance Chat</h2>', unsafe_allow_html=True)
        
        if not st.session_state.agent_initialized:
            st.markdown('<div class="warning-box">⚠️ Please initialize the agent first using the sidebar.</div>', unsafe_allow_html=True)
        else:
            # Chat interface
            chat_container = st.container()
            
            # Display chat history
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 GDPR Assistant:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input
            user_question = st.chat_input("Ask me anything about GDPR compliance...")
            
            if user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                try:
                    # Get agent response
                    with st.spinner("🤔 Analyzing your question..."):
                        agent = get_gdpr_agent()
                        result = agent.answer_gdpr_question(user_question)
                        
                        if 'error' in result:
                            response = f"I encountered an error: {result['error']}"
                        else:
                            response = result['answer']
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Rerun to display new messages
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab2:
        st.markdown('<h2 class="sub-header">🌐 Website Privacy Policy Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.agent_initialized:
            st.markdown('<div class="warning-box">⚠️ Please initialize the agent first using the sidebar.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                Enter a website URL to analyze its privacy policy for GDPR compliance.
                The AI will scrape the content and provide detailed compliance assessment.
            </div>
            """, unsafe_allow_html=True)
            
            # URL input
            col1, col2 = st.columns([3, 1])
            with col1:
                website_url = st.text_input(
                    "Website URL",
                    placeholder="https://example.com/privacy-policy",
                    help="Enter the full URL including https://"
                )
            
            with col2:
                analyze_button = st.button("🔍 Analyze", type="primary")
            
            if analyze_button and website_url:
                if not website_url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    try:
                        with st.spinner(f"🔍 Analyzing privacy policy at {website_url}..."):
                            agent = get_gdpr_agent()
                            analysis_result = agent.analyze_website_privacy_policy(website_url)
                            
                            if 'error' in analysis_result:
                                st.error(f"Analysis failed: {analysis_result['error']}")
                            else:
                                st.markdown('<div class="success-box">✅ Analysis completed!</div>', unsafe_allow_html=True)
                                
                                # Display results
                                st.markdown("### 📋 Analysis Results")
                                st.write(analysis_result['agent_analysis'])
                                
                                # Display relevant GDPR articles
                                if analysis_result.get('relevant_gdpr_articles'):
                                    st.markdown("### 📚 Relevant GDPR Articles")
                                    for i, article in enumerate(analysis_result['relevant_gdpr_articles'][:3], 1):
                                        with st.expander(f"Article {i}: {article['metadata'].get('article_title', 'GDPR Reference')}"):
                                            st.write(article['text'][:500] + "...")
                                
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 GDPR Violations Analytics</h2>', unsafe_allow_html=True)
        
        # Load and display analytics
        gdpr_text_df, violations_df = load_sample_data()
        
        if violations_df is not None:
            display_violation_analytics(violations_df)
        else:
            st.markdown('<div class="warning-box">⚠️ Violation data not available. Please ensure gdpr_violations.csv is in the project directory.</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">📚 GDPR Knowledge Base</h2>', unsafe_allow_html=True)
        
        if not st.session_state.agent_initialized:
            st.markdown('<div class="warning-box">⚠️ Please initialize the agent first using the sidebar.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                Search through the comprehensive GDPR knowledge base containing all articles,
                regulations, and real-world violation cases.
            </div>
            """, unsafe_allow_html=True)
            
            # Search interface
            search_query = st.text_input(
                "Search Knowledge Base",
                placeholder="e.g., data subject rights, lawful basis, consent...",
                help="Enter keywords to search through GDPR articles and violation cases"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                num_results = st.selectbox("Results", [5, 10, 15, 20], index=0)
            
            if st.button("🔍 Search Knowledge Base") and search_query:
                try:
                    with st.spinner("🔍 Searching knowledge base..."):
                        agent = get_gdpr_agent()
                        results = agent.search_knowledge_base(search_query, k=num_results)
                        
                        if results:
                            st.markdown(f"### 🎯 Found {len(results)} relevant results")
                            
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - {result['metadata']['type'].replace('_', ' ').title()} (Score: {result['similarity_score']:.3f})"):
                                    if result['metadata']['type'] == 'gdpr_text':
                                        st.markdown(f"**Chapter {result['metadata']['chapter']}:** {result['metadata']['chapter_title']}")
                                        st.markdown(f"**Article {result['metadata']['article']}:** {result['metadata']['article_title']}")
                                        st.markdown(f"**Sub-article:** {result['metadata']['sub_article']}")
                                        if result['metadata'].get('href'):
                                            st.markdown(f"**Reference:** [{result['metadata']['href']}]({result['metadata']['href']})")
                                    else:  # violation_case
                                        st.markdown(f"**Country:** {result['metadata']['country']}")
                                        st.markdown(f"**Fine:** €{result['metadata']['fine_amount']:,}")
                                        st.markdown(f"**Authority:** {result['metadata']['authority']}")
                                        st.markdown(f"**Date:** {result['metadata']['date']}")
                                        st.markdown(f"**Articles Violated:** {result['metadata']['articles_violated']}")
                                    
                                    st.markdown("**Content:**")
                                    st.write(result['text'])
                        else:
                            st.warning("No results found. Try different keywords.")
                            
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🔒 GDPR Compliance Assistant | Powered by aiXplain & Streamlit<br>
        <small>Built for aiXplain Platform Certification</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
