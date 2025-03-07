import os
import json
import streamlit as st
from flow import online_flow
from utils.vector_search import load_index
import base64
import logging
from PIL import Image
import io
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("paul_graham_streamlit")

# Define audio cache directory
AUDIO_CACHE_DIR = "audio_cache"

# Set up page config
st.set_page_config(
    page_title="AI Paul Graham",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
    }
    .subheader {
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .stSpinner {
        margin-bottom: 10px;
    }
    .log-container {
        background-color: #1E1E1E;
        color: #DCDCDC;
        font-family: monospace;
        padding: 10px;
        border-radius: 5px;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system(output_dir="output"):
    """Load necessary resources at system startup."""
    with st.spinner("Loading Paul Graham's knowledge base..."):
        # Load FAISS index
        faiss_index_path = os.path.join(output_dir, "essay_index.faiss")
        faiss_index = load_index(faiss_index_path)
        
        # Load chunk metadata (includes text content)
        metadata_path = os.path.join(output_dir, "chunk_metadata.json")
        with open(metadata_path, "r") as f:
            chunk_metadata = json.load(f)
        
        # Load essay metadata from CSV
        meta_csv_path = "meta.csv"
        if os.path.exists(meta_csv_path):
            essay_metadata = pd.read_csv(meta_csv_path)
            # Convert to dictionary for easier lookup by essay_id
            essay_metadata_dict = essay_metadata.set_index('text_id').to_dict(orient='index')
        else:
            logger.warning(f"Meta CSV file not found at {meta_csv_path}")
            essay_metadata_dict = {}
        
        st.session_state.system_resources = {
            "faiss_index": faiss_index,
            "chunk_metadata": chunk_metadata,
            "essay_metadata": essay_metadata_dict
        }
        
        return st.session_state.system_resources

def get_audio_player(audio_path):
    """Create an HTML5 audio player for the given audio file."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_player = f"""
        <audio controls autoplay class="stAudio">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    """
    return audio_player

def main():
    # Initialize processing status in session state if not present
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    if "submitted_query" not in st.session_state:
        st.session_state.submitted_query = ""
        
    # Function to handle form submission
    def handle_submission():
        if not st.session_state.user_query.strip():
            st.error("Please enter your question.")
            return
        elif len(st.session_state.user_query.strip()) < 10:
            st.error("Your question should be at least 10 characters long.")
            return
        else:
            st.session_state.is_processing = True
            st.session_state.submitted_query = st.session_state.user_query

    # Initialize system if not already done
    if 'system_resources' not in st.session_state:
        system_resources = initialize_system()
    else:
        system_resources = st.session_state.system_resources
    

    # Put text area + submit button together in a form
    with st.form("ask_paul_form"):
        # Display header
        st.markdown('<p class="main-header">Ask AI Paul Graham</p>', unsafe_allow_html=True)

        # Add Paul Graham's image with round corners and shadow
        # Load the image from the assets directory
        image_path = os.path.join("assets", "paul_graham.png")
        if os.path.exists(image_path):
            # Apply custom CSS for the circular image with shadow
            st.markdown("""
            <style>
            .circular-image {
                display: flex;
                justify-content: center;
                margin-bottom: 30px;
            }
            .circular-image img {
                width: 150px;
                height: 150px;
                border-radius: 50%;
                object-fit: cover;
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
                border: 3px solid #ffffff;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create columns to center the image
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                # Use HTML for the image to maintain styling control
                img = Image.open(image_path)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
                
                st.markdown(f"""
                <div class="circular-image">
                    <img src="data:image/png;base64,{img_base64}" alt="Paul Graham">
                </div>
                """, unsafe_allow_html=True) 
                
        user_query = st.text_area(
            "What would you like to ask Paul Graham?",
            value="I sent 50 cold emails to potential customers, but no one responded. I feel like my life is a failure. What should I do?",
            key="user_query",
            height=100, 
            max_chars=500,
            placeholder="e.g., What advice do you have for startup founders?",
        )
        
        st.markdown('<div style="font-size: 0.8em; color: #666; margin-bottom: 20px;">This project is fully open sourced at: <a href="https://github.com/The-Pocket/Tutorial-YC-Partner">GitHub</a><br>This is an example LLM project for <a href="https://github.com/The-Pocket/PocketFlow">Pocket Flow</a>, a 100-line LLM framework.</div>', unsafe_allow_html=True)

        # This button will submit the form (capturing your most recent text)
        submitted = st.form_submit_button(
            "Ask Paul", 
            disabled=st.session_state.is_processing, 
            type="primary", 
            use_container_width=True,
            on_click=handle_submission
        )

    # Process the query if we're in processing state
    if st.session_state.is_processing and st.session_state.submitted_query:
        # Preserve the submitted query for display
        user_query = st.session_state.submitted_query
        
        # Create shared data for this query
        shared = {
            # System resources
            "faiss_index": system_resources["faiss_index"],
            "chunk_metadata": system_resources["chunk_metadata"],
            "essay_metadata": system_resources["essay_metadata"],
            
            # Query
            "query": user_query
        }
        
        # Show progress with live logging
        with st.spinner("Paul is thinking..."):
            # Create a status area and log area
            status_area = st.empty()
            log_area = st.empty()
            
            # Create a custom log handler to display logs in real-time
            log_messages = []
            class StreamlitLogHandler(logging.Handler):
                def emit(self, record):
                    log_messages.append(self.format(record))
                    log_text = "\n".join(log_messages[-10:])  # Show last 10 messages
                    log_area.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)
            
            # Add the custom handler to the logger
            streamlit_handler = StreamlitLogHandler()
            streamlit_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logging.getLogger().addHandler(streamlit_handler)
            
            try:
                # Run the online processing flow
                online_flow.run(shared)
                # Don't show "Processing completed" message
                status_area.empty()
                
                # Store a copy of the shared data for debugging
                st.session_state.last_query_data = shared.copy()
                
                # Remove the custom handler
                logging.getLogger().removeHandler(streamlit_handler)
                
            except Exception as e:
                status_area.error(f"An error occurred: {str(e)}")
                logging.getLogger().removeHandler(streamlit_handler)
                st.error(f"Failed to generate response: {str(e)}")
                # Clear submitted query to prevent reprocessing on page refresh
                st.session_state.submitted_query = ""
                return
            
            # Clear the live log display after processing
            log_area.empty()
        
        # Get the response
        if not shared.get("is_valid_query", True):
            response = shared['final_response']
        else:
            if isinstance(shared['final_response'], dict) and 'content' in shared['final_response']:
                # Extract just the content if it's a dictionary
                response = shared['final_response']['content']
            else:
                response = shared['final_response']
            
        # Remove "humm" occurrences from the response
        response = response.replace(" Humm.", "").replace("Humm. ", "").replace("Humm.", "").replace("Humm,", "")
        
        # Extract and deduplicate essay sources if available
        useful_resources = []
        if 'relevant_chunks' in shared and shared['relevant_chunks']:
            essay_ids = set()
            for chunk in shared['relevant_chunks']:
                if chunk.get('is_relevant', False):
                    essay_id = chunk['metadata'].get('essay_id', 'unknown')
                    # Try to extract the numeric essay_id
                    try:
                        # Some essay_ids might be in the format "essay_123", try to extract the number
                        if isinstance(essay_id, str) and "_" in essay_id:
                            essay_id = int(essay_id.split("_")[1])
                        else:
                            essay_id = int(essay_id)
                        essay_ids.add(essay_id)
                    except (ValueError, TypeError):
                        # If we can't convert to int, skip this essay_id
                        logger.warning(f"Could not convert essay_id {essay_id} to integer")
                        continue
            
            # Get essay metadata for these IDs
            essay_metadata = shared.get('essay_metadata', {})
            for essay_id in sorted(essay_ids):
                if essay_id in essay_metadata:
                    metadata = essay_metadata[essay_id]
                    title = metadata.get('title', f"Essay {essay_id}")
                    link = metadata.get('link', '')
                    useful_resources.append({
                        'essay_id': essay_id,
                        'title': title,
                        'link': link
                    })

        # Display the response
        st.markdown(f"**Paul Graham:** {response}")

        # Check if audio is available
        audio_hash = shared.get('audio_file_hash')
        if audio_hash:
            audio_path = os.path.join(AUDIO_CACHE_DIR, f"{audio_hash}.wav")
            if os.path.exists(audio_path):
                st.markdown(get_audio_player(audio_path), unsafe_allow_html=True)
        
        # Display useful resources if available
        if useful_resources:
            st.markdown("**Useful Resources:**", unsafe_allow_html=True)
            for resource in useful_resources:
                title = resource['title']
                link = resource['link']
                if link:
                    st.markdown(f"- [{title}]({link})", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {title}", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 