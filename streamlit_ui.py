"""Streamlit UI for AI Ad Generation System."""

import base64
import logging
import time
from io import BytesIO
from typing import Dict, Any

import requests
import streamlit as st
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Ad Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE = "http://localhost:8000"
TIMEOUT_SECONDS = 60

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stat-metric {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, method: str = "GET", **kwargs) -> tuple[bool, Any]:
    """Make API request with error handling.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        **kwargs: Additional arguments for requests
        
    Returns:
        Tuple of (success, response_data)
    """
    try:
        url = f"{API_BASE}{endpoint}"
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT_SECONDS, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, timeout=TIMEOUT_SECONDS, **kwargs)
        else:
            return False, f"Unsupported HTTP method: {method}"
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", str(error_data))
            except:
                error_detail = response.text or f"HTTP {response.status_code}"
            
            return False, error_detail
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API server. Is it running?"
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return False, f"Request failed: {str(e)}"


def check_api_health() -> tuple[bool, Dict[str, Any]]:
    """Check API health status."""
    return make_api_request("/health")


def display_health_status(health_data: Dict[str, Any]):
    """Display health status information."""
    status = health_data.get("api_status", "unknown")
    
    if status == "healthy":
        st.markdown('<div class="success-box">✅ API Server: Healthy</div>', 
                   unsafe_allow_html=True)
    elif status == "degraded":
        st.markdown('<div class="error-box">⚠️ API Server: Degraded</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ API Server: Unhealthy</div>', 
                   unsafe_allow_html=True)
    
    # Show component status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vectorizer_status = "✅" if health_data.get("vectorizer_initialized") else "❌"
        st.metric("Vectorizer", vectorizer_status)
    
    with col2:
        agent_status = "✅" if health_data.get("agent_initialized") else "❌"
        st.metric("Agent", agent_status)
    
    with col3:
        doc_count = health_data.get("document_count", 0)
        st.metric("Documents", doc_count)
    
    # Show errors if any
    errors = health_data.get("errors", [])
    if errors:
        st.markdown("**Configuration Issues:**")
        for error in errors:
            st.markdown(f"- ⚠️ {error}")


def display_messages(messages):
    """Display process messages."""
    if not messages:
        return
    
    st.markdown("**Process Log:**")
    for msg in messages:
        msg_type = msg.get("type", "info")
        content = msg.get("content", "")
        
        if msg_type == "success":
            st.success(content)
        elif msg_type == "error":
            st.error(content)
        elif msg_type == "warning":
            st.warning(content)
        else:
            st.info(content)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Ad Generation System</h1>', 
                unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.markdown(f'<div class="error-box">❌ Cannot connect to API server: {health_data}</div>', 
                   unsafe_allow_html=True)
        st.markdown("**To start the API server, run:**")
        st.code("uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")
        display_health_status(health_data)
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Settings
        st.markdown("### Settings")
        max_results = st.slider("Max Search Results", 1, 20, 5)
        
        # Statistics
        success, stats_data = make_api_request("/stats/")
        if success:
            st.markdown("### Statistics")
            vector_store = stats_data.get("vector_store", {})
            if vector_store.get("exists"):
                st.metric("Total Chunks", vector_store.get("count", 0))
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload Documents", 
        "🎨 Generate Ads", 
        "🔍 Search Knowledge", 
        "📊 System Stats"
    ])
    
    # Tab 1: Upload Documents
    with tab1:
        st.header("📁 Upload PDF Documents")
        st.markdown("""
        Upload PDF documents to build your knowledge base for ad generation. 
        These documents will be processed and used to provide context for creating better advertisements.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload PDF documents containing advertising guidelines, brand information, or design principles."
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"File: {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            if st.button("📤 Upload and Process", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    success, response = make_api_request("/upload-pdf/", "POST", files=files)
                    
                    if success:
                        st.success(f"✅ {response['message']}")
                        st.info(f"Processed {response['processed_chunks']} document chunks")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response}")
    
    # Tab 2: Generate Ads
    with tab2:
        st.header("🎨 Generate Advertisement")
        st.markdown("""
        Describe the advertisement you want to create. The AI will use your uploaded documents 
        and professional advertising principles to generate a compelling visual advertisement.
        """)
        
        # Ad generation form
        with st.form("ad_generation_form"):
            user_request = st.text_area(
                "Describe your advertisement:",
                height=150,
                placeholder="Example: Create a modern advertisement for a sustainable coffee brand targeting millennials, emphasizing eco-friendly packaging and premium quality...",
                help="Be specific about your product, target audience, style preferences, and key messages."
            )
            
            submitted = st.form_submit_button("🎨 Generate Advertisement", use_container_width=True)
        
        if submitted and user_request.strip():
            with st.spinner("Generating your advertisement..."):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Retrieving relevant context...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🧠 Generating ad concept with AI...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("🎨 Creating visual advertisement...")
                progress_bar.progress(75)
                
                # Make API request
                success, response = make_api_request(
                    "/generate-ad/", 
                    "POST", 
                    json={"user_request": user_request}
                )
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if success:
                    st.markdown("### Generated Advertisement")
                    
                    # Show the generated image
                    if response.get("generated_image"):
                        try:
                            image_data = base64.b64decode(response["generated_image"])
                            image = Image.open(BytesIO(image_data))
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(image, caption="Generated Advertisement", use_column_width=True)
                            
                            with col2:
                                st.markdown("**Download Options:**")
                                
                                # Download button
                                btn = st.download_button(
                                    label="💾 Download Image",
                                    data=image_data,
                                    file_name="generated_ad.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                                
                                # Image info
                                st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                                
                        except Exception as e:
                            st.error(f"Failed to display image: {e}")
                    
                    # Show the prompt used
                    if response.get("ad_prompt"):
                        with st.expander("🔍 View Generated Prompt"):
                            st.text_area(
                                "AI-generated prompt used for image creation:",
                                value=response["ad_prompt"],
                                height=100,
                                disabled=True
                            )
                    
                    # Show process messages
                    if response.get("messages"):
                        with st.expander("📋 Process Details"):
                            display_messages(response["messages"])
                    
                    # Context indicator
                    if response.get("context_used"):
                        st.success("✅ Used context from your uploaded documents")
                    else:
                        st.info("ℹ️ Used default advertising principles (no documents uploaded)")
                
                else:
                    st.error(f"❌ Generation failed: {response}")
        
        elif submitted:
            st.warning("⚠️ Please enter a description for your advertisement.")
    
    # Tab 3: Search Knowledge
    with tab3:
        st.header("🔍 Search Knowledge Base")
        st.markdown("Search through your uploaded documents to find relevant information.")
        
        # Search form
        search_query = st.text_input(
            "Search query:",
            placeholder="Enter keywords to search your documents...",
            help="Search for specific terms, concepts, or topics in your uploaded documents."
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            search_clicked = st.button("🔍 Search", use_container_width=True)
        
        if search_clicked and search_query.strip():
            with st.spinner("Searching documents..."):
                success, response = make_api_request(
                    "/search/", 
                    "GET", 
                    params={"query": search_query, "k": max_results}
                )
                
                if success:
                    results = response.get("results", [])
                    total = response.get("total_results", 0)
                    
                    if results:
                        st.success(f"✅ Found {total} results for '{search_query}'")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"📄 Result {i} - {result.get('source', 'Unknown Source')}"):
                                st.write(result.get('content', 'No content'))
                                
                                if result.get('source'):
                                    st.caption(f"Source: {result['source']}")
                    else:
                        st.info(f"No results found for '{search_query}'")
                        st.markdown("**Suggestions:**")
                        st.markdown("- Try different keywords")
                        st.markdown("- Upload more documents to expand your knowledge base")
                        st.markdown("- Use broader search terms")
                
                else:
                    st.error(f"❌ Search failed: {response}")
        
        elif search_clicked:
            st.warning("⚠️ Please enter a search query.")
    
    # Tab 4: System Statistics
    with tab4:
        st.header("📊 System Statistics")
        
        # Refresh stats
        if st.button("🔄 Refresh Statistics"):
            st.rerun()
        
        success, stats_data = make_api_request("/stats/")
        
        if success:
            # System Status
            st.subheader("System Status")
            status = stats_data.get("system_status", "unknown")
            
            if status == "operational":
                st.success("✅ System is operational")
            else:
                st.warning(f"⚠️ System status: {status}")
            
            # Component Status
            st.subheader("Components")
            components = stats_data.get("components", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                vectorizer_status = "✅ Active" if components.get("vectorizer") else "❌ Inactive"
                st.metric("PDF Vectorizer", vectorizer_status)
            
            with col2:
                agent_status = "✅ Active" if components.get("agent") else "❌ Inactive"
                st.metric("AI Agent", agent_status)
            
            # Configuration
            st.subheader("Configuration")
            config = stats_data.get("configuration", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Upload Directory", config.get("upload_directory", "N/A"))
            
            with col2:
                st.metric("Vector Store Path", config.get("vector_store_path", "N/A"))
            
            with col3:
                max_size = config.get("max_file_size_mb", "N/A")
                st.metric("Max File Size (MB)", max_size)
            
            # Vector Store Stats
            vector_store = stats_data.get("vector_store", {})
            if vector_store and not vector_store.get("error"):
                st.subheader("Knowledge Base")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    exists = "Yes" if vector_store.get("exists") else "No"
                    st.metric("Vector Store Exists", exists)
                
                with col2:
                    count = vector_store.get("count", 0)
                    st.metric("Document Chunks", count)
                
                with col3:
                    collection = vector_store.get("collection_name", "N/A")
                    st.metric("Collection Name", collection)
                
                if count > 0:
                    st.success(f"✅ Knowledge base contains {count} document chunks")
                else:
                    st.info("ℹ️ No documents uploaded yet")
            
            # Agent Stats
            agent_info = stats_data.get("agent", {})
            if agent_info and not agent_info.get("error"):
                st.subheader("AI Agent Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    gemini_status = "✅ Available" if agent_info.get("gemini_available") else "❌ Unavailable"
                    st.metric("Gemini AI", gemini_status)
                    
                    model_name = agent_info.get("model_name", "N/A")
                    st.metric("Gemini Model", model_name)
                
                with col2:
                    img_gen_status = "✅ Available" if agent_info.get("image_generator_available") else "❌ Unavailable"
                    st.metric("Image Generator", img_gen_status)
                    
                    flux_model = agent_info.get("flux_model", "N/A")
                    st.metric("FLUX Model", flux_model)
            
            # Health Check Data
            if health_data.get("errors"):
                st.subheader("⚠️ Configuration Issues")
                for error in health_data["errors"]:
                    st.error(error)
        
        else:
            st.error(f"❌ Failed to load statistics: {stats_data}")
        
        # Performance Metrics (if available)
        st.subheader("Performance")
        
        # Test API response time
        if st.button("🏃 Test API Response Time"):
            start_time = time.time()
            success, _ = check_api_health()
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            
            if success:
                if response_time < 100:
                    st.success(f"✅ API response time: {response_time:.2f} ms (Excellent)")
                elif response_time < 500:
                    st.info(f"ℹ️ API response time: {response_time:.2f} ms (Good)")
                else:
                    st.warning(f"⚠️ API response time: {response_time:.2f} ms (Slow)")
            else:
                st.error("❌ API not responding")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Streamlit app error: {e}")
        st.error(f"Application error: {e}")
        st.markdown("**Please check the logs and try refreshing the page.**")