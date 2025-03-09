import streamlit as st
import os
from pdf_processor import extract_text_from_pdf, clean_text
from embedding import create_embeddings, search_similar_chunks
from rag import generate_response
from utils import save_uploaded_file, get_temp_file_path, format_bytes
from streamlit_compat import divider, rerun
from metadata_extractor import MetadataExtractor
from summary_generator import SummaryGenerator
import time
from typing import List, Dict
import re

# Initialize metadata extractor and summary generator
metadata_extractor = MetadataExtractor()
summary_generator = SummaryGenerator()

# Configure page
st.set_page_config(
    page_title="Legal Document Analyzer with Private LLM",
    page_icon="📜",
    layout="wide"
)

# Apply custom CSS for cursor behavior and enhanced UI
custom_css = """
<style>
/* Basic cursor rules */
button:hover, .stButton>button:hover {
    cursor: pointer !important;
}
.stTextInput > div > input, .stTextArea > div > textarea {
    cursor: text !important;
}
.stFileUploader, [data-testid="stFileDropzone"] {
    cursor: copy !important;
}
.disabled {
    cursor: not-allowed !important;
}
.stSpinner {
    cursor: progress !important;
}
a:hover {
    cursor: pointer !important;
}

/* Enhanced UI styling */
.document-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
    background-color: #f9f9f9;
}
.document-card:hover {
    border-color: #4a80b8;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metadata-item {
    padding: 3px 0;
}
.card-title {
    font-weight: bold;
    color: #2c3e50;
}
.section-header {
    background-color: #f1f6ff;
    padding: 5px 10px;
    border-left: 4px solid #4a80b8;
    margin: 10px 0;
}

/* ENHANCED TAB STYLING */
.stTabs {
    background-color: #121212;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    padding: 3px;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #333;
    padding-bottom: 0;
    display: flex;
    justify-content: flex-start;
}

.stTabs [data-baseweb="tab"] {
    height: 54px;
    white-space: pre-wrap;
    background-color: rgba(60, 60, 60, 0.3);
    border-radius: 8px 8px 0 0;
    gap: 1px;
    padding: 12px 20px;
    font-weight: 500;
    color: #adb5bd;
    margin-right: 0px;
    border-right: 1px solid #333;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(80, 80, 80, 0.5);
    color: #f8f9fa;
}

.stTabs [aria-selected="true"] {
    background-color: #1e3d59 !important;
    color: #ffffff !important;
    font-weight: 600;
    border-bottom: 3px solid #4a80b8;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Tab content area */
.stTabs [data-testid="stTabContent"] {
    background-color: #121212;
    padding: 15px;
    border-radius: 0 0 8px 8px;
    border: 1px solid #333;
    border-top: none;
}

/* Add icons using before pseudo-element */
.stTabs [data-baseweb="tab"]:nth-child(1):before {
    content: "💬 ";
}

.stTabs [data-baseweb="tab"]:nth-child(2):before {
    content: "📋 ";
}

.stTabs [data-baseweb="tab"]:nth-child(3):before {
    content: "📝 ";
}

.stTabs [data-baseweb="tab"]:nth-child(4):before {
    content: "📄 ";
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state for multiple documents
if "documents" not in st.session_state:
    st.session_state.documents = {}  # Store document info by document_id
if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "combined_chunks" not in st.session_state:
    st.session_state.combined_chunks = []
if "combined_embeddings" not in st.session_state:
    st.session_state.combined_embeddings = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

# Function to clear chat history
def clear_chat():
    st.session_state.chat_history = []

# Function to process document and extract metadata/summary
def process_document(file_path, filename, file_size):
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(file_path)
    
    # Clean the text
    cleaned_text = clean_text(extracted_text)
    
    # Extract metadata
    with st.spinner("Extracting metadata..."):
        metadata = metadata_extractor.extract_metadata(cleaned_text, filename)
    
    # Generate summary
    with st.spinner("Generating summary..."):
        summary_data = summary_generator.generate_summary(cleaned_text, metadata)
    
    # Create chunks and embeddings
    try:
        with st.spinner("Creating embeddings..."):
            chunks, embeddings = create_embeddings(cleaned_text)
            
        return {
            "filename": filename,
            "text": cleaned_text,
            "chunks": chunks,
            "embeddings": embeddings,
            "file_size": file_size,
            "metadata": metadata,
            "summary": summary_data["summary"],
            "key_points": summary_data["key_points"],
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return {
            "filename": filename,
            "text": cleaned_text,
            "chunks": [],
            "embeddings": None,
            "file_size": file_size,
            "metadata": metadata,
            "summary": summary_data["summary"],
            "key_points": summary_data["key_points"],
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# Enhanced Response Deduplication and Formatting

def process_llm_response(response: str) -> str:
    """Process and clean up LLM response with enhanced deduplication."""
    # If the response is just repeating the same sentence multiple times
    if response.count('.') >= 2:
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        unique_sentences = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Check if this sentence is essentially the same as any we've already included
            is_duplicate = False
            for existing in unique_sentences:
                # Compare sentence similarity - if they're 80% the same, consider it a duplicate
                if _similarity_ratio(sentence.lower(), existing.lower()) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        # Limit to 3 sentences max
        unique_sentences = unique_sentences[:3]
        response = ' '.join(unique_sentences)
    
    # Ensure first letter is capitalized and there's punctuation at the end
    if response:
        response = response[0].upper() + response[1:]
        if not response[-1] in '.!?':
            response += '.'
            
    return response

def _similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    # Simple implementation - can be replaced with difflib or other similarity measures
    shorter = min(len(s1), len(s2))
    longer = max(len(s1), len(s2))
    
    if longer == 0:
        return 1.0
        
    # Count matching characters
    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    return matches / longer

# Title and description
st.title("Legal Document Analyzer")
st.markdown("Upload legal documents, extract insights, and chat with an AI to understand their content.")

# Sidebar for file upload and processing
with st.sidebar:
    st.header("Document Upload")
    
    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        # Display files info
        st.subheader("Uploaded Files")
        for uploaded_file in uploaded_files:
            doc_id = f"{uploaded_file.name}_{hash(uploaded_file.name)}"
            
            # Check if this file was already processed
            already_processed = doc_id in st.session_state.documents
            
            # Display file with processed status
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{uploaded_file.name} ({format_bytes(uploaded_file.size)})")
            with col2:
                if already_processed:
                    st.success("✓")
                else:
                    st.warning("⌛")
        
        # Process all button
        if st.button("Process All Documents"):
            with st.spinner("Processing documents..."):
                all_chunks = []
                all_embeddings_list = []
                
                for uploaded_file in uploaded_files:
                    doc_id = f"{uploaded_file.name}_{hash(uploaded_file.name)}"
                    
                    # Skip if already processed
                    if doc_id in st.session_state.documents:
                        all_chunks.extend(st.session_state.documents[doc_id]["chunks"])
                        continue
                    
                    # Save the uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Process the document
                    doc_info = process_document(file_path, uploaded_file.name, uploaded_file.size)
                    
                    # Store in session state
                    st.session_state.documents[doc_id] = doc_info
                    
                    # Add to combined data if embeddings exist
                    if doc_info["embeddings"] is not None and len(doc_info["chunks"]) > 0:
                        all_chunks.extend(doc_info["chunks"])
                        all_embeddings_list.append(doc_info["embeddings"])
                
                # Combine embeddings if we have any
                if all_embeddings_list:
                    import numpy as np
                    try:
                        st.session_state.combined_chunks = all_chunks
                        st.session_state.combined_embeddings = np.vstack(all_embeddings_list)
                        st.success(f"Successfully processed {len(st.session_state.documents)} documents with {len(all_chunks)} total chunks")
                    except Exception as e:
                        st.error(f"Error combining embeddings: {str(e)}")
    
    # Document selection
    if st.session_state.documents:
        st.subheader("Select Document")
        doc_options = ["All Documents"] + [doc["filename"] for doc in st.session_state.documents.values()]
        selected_doc = st.selectbox("Choose a document:", doc_options)
        
        if selected_doc != "All Documents":
            # Find the selected document id
            for doc_id, doc in st.session_state.documents.items():
                if doc["filename"] == selected_doc:
                    st.session_state.selected_document_id = doc_id
                    break
        else:
            st.session_state.selected_document_id = "all"
    
    divider()
    
    # Add options for RAG settings
    st.subheader("RAG Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    
    # Option to include all documents or just the selected one
    use_all_docs = st.checkbox("Search across all documents", value=True)
    
    # Add clear button
    if st.button("Clear Chat History"):
        clear_chat()
    
    divider()
    
    # Document comparison feature
    if len(st.session_state.documents) >= 2:
        st.subheader("Document Comparison")
        if st.button("Compare Documents"):
            with st.spinner("Generating document comparison..."):
                # Get the documents to compare (limit to first 3)
                docs_to_compare = list(st.session_state.documents.values())[:3]
                doc_texts = [doc["text"] for doc in docs_to_compare]
                doc_titles = [doc["filename"] for doc in docs_to_compare]
                
                # Generate comparison
                comparison = summary_generator.generate_document_comparison(doc_texts, doc_titles)
                
                # Create a new tab with the comparison
                st.session_state.comparison_result = comparison
                st.session_state.active_tab = "Comparison"
                
                # Rerun to update UI
                rerun()

# Main area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Metadata", "Summary", "Comparison"])

# Chat tab
with tab1:
    if st.session_state.active_tab == "Chat":
        st.session_state.active_tab = "Chat"  # Ensure we stay on this tab
    
    col1, col2 = st.columns([2, 3])
    
    # Document preview column
    with col1:
        st.header("Document Preview")
        
        if st.session_state.documents and st.session_state.selected_document_id:
            if st.session_state.selected_document_id == "all":
                # Preview of all documents (just show a summary)
                st.info(f"You have {len(st.session_state.documents)} documents processed with {len(st.session_state.combined_chunks)} total chunks.")
                
                # Display a list of all documents
                st.subheader("Processed Documents")
                for doc_id, doc in st.session_state.documents.items():
                    with st.expander(f"{doc['filename']} ({format_bytes(doc['file_size'])})"):
                        st.write(f"**Type:** {doc['metadata'].get('contract_type', doc['metadata'].get('document_type', 'Unknown'))}")
                        st.write(f"**Processed:** {doc['processed_date']}")
                        st.write(f"**Pages (est.):** {doc['metadata'].get('estimated_page_count', 'Unknown')}")
            else:
                # Show the selected document
                doc = st.session_state.documents[st.session_state.selected_document_id]
                st.text_area(f"Content of {doc['filename']}", doc['text'], height=500)
        else:
            st.info("No documents processed yet. Please upload PDFs and click 'Process All Documents'.")
    
    # Chat interface column
    with col2:
        st.header("Chat with Documents")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"💬 **You**: {message['content']}")
                else:
                    st.markdown(f"🤖 **AI**: {message['content']}")
                
                # Add a small divider except after the last message
                if i < len(st.session_state.chat_history) - 1:
                    divider()
        
        # Query input
        st.subheader("Ask a question")
        with st.form(key="query_form"):
            query = st.text_area("Enter your question about the documents:", height=100)
            submit_button = st.form_submit_button("Send")
            
            if submit_button and query:
                if not st.session_state.documents:
                    st.error("Please process at least one document first!")
                else:
                    # Add user query to chat history
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    
                    # Determine which chunks and embeddings to use
                    chunks_to_search = []
                    embeddings_to_search = None
                    
                    if use_all_docs:
                        # Use combined data from all documents
                        chunks_to_search = st.session_state.combined_chunks
                        embeddings_to_search = st.session_state.combined_embeddings
                    elif st.session_state.selected_document_id and st.session_state.selected_document_id != "all":
                        # Use only the selected document
                        doc = st.session_state.documents[st.session_state.selected_document_id]
                        chunks_to_search = doc["chunks"]
                        embeddings_to_search = doc["embeddings"]
                    else:
                        # Default to all documents
                        chunks_to_search = st.session_state.combined_chunks
                        embeddings_to_search = st.session_state.combined_embeddings
                    
                    # Find relevant chunks or use simple approach if embeddings failed
                    if embeddings_to_search is not None and len(chunks_to_search) > 0:
                        # Find relevant chunks
                        with st.spinner("Searching documents..."):
                            relevant_chunks = search_similar_chunks(
                                query, 
                                chunks_to_search, 
                                embeddings_to_search, 
                                k=top_k
                            )
                    else:
                        # Use a simple approach if embeddings failed
                        if st.session_state.documents:
                            # Get text from the first document as fallback
                            first_doc = next(iter(st.session_state.documents.values()))
                            relevant_chunks = [first_doc["text"][:5000]]
                        else:
                            st.error("No document content available")
                            relevant_chunks = ["No document content available"]
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = generate_response(query, relevant_chunks, st.session_state.documents)
                    
                    # Apply post-processing
                    response = process_llm_response(response)
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the UI
                    rerun()

# Metadata tab
with tab2:
    if st.session_state.active_tab == "Metadata":
        st.session_state.active_tab = "Metadata"  # Ensure we stay on this tab
    
    st.header("Document Metadata")
    
    if st.session_state.documents:
        if st.session_state.selected_document_id and st.session_state.selected_document_id != "all":
            # Show metadata for selected document
            doc = st.session_state.documents[st.session_state.selected_document_id]
            
            st.subheader(f"Metadata for {doc['filename']}")
            
            if doc.get("metadata"):
                metadata = doc["metadata"]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**Title:** {metadata.get('title', 'Untitled')}")
                    st.markdown(f"**Type:** {metadata.get('contract_type', metadata.get('document_type', 'Unknown'))}")
                    st.markdown(f"**Estimated Pages:** {metadata.get('estimated_page_count', 'Unknown')}")
                    
                    st.markdown("**Important Dates:**")
                    if metadata.get('dates'):
                        for date in metadata.get('dates')[:3]:
                            st.markdown(f"- {date}")
                
                with col2:
                    st.markdown("**Organizations:**")
                    if metadata.get('organizations'):
                        for org in metadata.get('organizations')[:3]:
                            if isinstance(org, dict):
                                org_name = org.get("name", str(org))
                                org_type = org.get("type", "")
                                st.markdown(f"- {org_name} ({org_type})" if org_type else f"- {org_name}")
                            else:
                                st.markdown(f"- {org}")
                    
                    st.markdown("**People:**")
                    if metadata.get('people'):
                        for person in metadata.get('people')[:3]:
                            if isinstance(person, dict):
                                person_name = person.get("name", str(person))
                                person_role = person.get("role", "")
                                st.markdown(f"- {person_name} ({person_role})" if person_role else f"- {person_name}")
                            else:
                                st.markdown(f"- {person}")
                    
                    st.markdown("**Monetary Values:**")
                    if metadata.get('monetary_values'):
                        for value in metadata.get('monetary_values')[:3]:
                            st.markdown(f"- {value}")
            else:
                st.info("No metadata available for this document.")
        else:
            # Show metadata for all documents
            for doc_id, doc in st.session_state.documents.items():
                with st.expander(f"{doc['filename']}"):
                    if doc.get("metadata"):
                        metadata = doc["metadata"]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"**Title:** {metadata.get('title', 'Untitled')}")
                            st.markdown(f"**Type:** {metadata.get('contract_type', metadata.get('document_type', 'Unknown'))}")
                            st.markdown(f"**Estimated Pages:** {metadata.get('estimated_page_count', 'Unknown')}")
                            
                            st.markdown("**Important Dates:**")
                            if metadata.get('dates'):
                                for date in metadata.get('dates')[:3]:
                                    st.markdown(f"- {date}")
                        
                        with col2:
                            st.markdown("**Organizations:**")
                            if metadata.get('organizations'):
                                for org in metadata.get('organizations')[:3]:
                                    if isinstance(org, dict):
                                        org_name = org.get("name", str(org))
                                        org_type = org.get("type", "")
                                        st.markdown(f"- {org_name} ({org_type})" if org_type else f"- {org_name}")
                                    else:
                                        st.markdown(f"- {org}")
                            
                            st.markdown("**People:**")
                            if metadata.get('people'):
                                for person in metadata.get('people')[:3]:
                                    if isinstance(person, dict):
                                        person_name = person.get("name", str(person))
                                        person_role = person.get("role", "")
                                        st.markdown(f"- {person_name} ({person_role})" if person_role else f"- {person_name}")
                                    else:
                                        st.markdown(f"- {person}")
                            
                            st.markdown("**Monetary Values:**")
                            if metadata.get('monetary_values'):
                                for value in metadata.get('monetary_values')[:3]:
                                    st.markdown(f"- {value}")
    else:
        st.info("No documents processed yet. Please upload PDFs and click 'Process All Documents'.")

# Summary tab
with tab3:
    if st.session_state.active_tab == "Summary":
        st.session_state.active_tab = "Summary"  # Ensure we stay on this tab
    
    st.header("Document Summaries")
    
    if st.session_state.documents:
        if st.session_state.selected_document_id and st.session_state.selected_document_id != "all":
            # Show summary for selected document
            doc = st.session_state.documents[st.session_state.selected_document_id]
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader(f"Summary of {doc['filename']}")
                st.markdown(doc["summary"])
            
            with col2:
                st.subheader("Key Points")
                st.markdown(doc["key_points"])
        else:
            # Show summaries for all documents
            for doc_id, doc in st.session_state.documents.items():
                with st.expander(f"{doc['filename']}"):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("Summary")
                        st.markdown(doc["summary"])
                    
                    with col2:
                        st.subheader("Key Points")