# PDF Chat with SambaNova Llama 3.1

This application allows users to upload PDF documents, process them, and interact with the content through a chat interface powered by SambaNova's Llama 3.1 model and Retrieval-Augmented Generation (RAG).

## Features

- PDF document upload and text extraction
- Text cleaning and processing
- Embedding generation and semantic search
- Retrieval-Augmented Generation (RAG)
- Interactive chat interface with custom cursor behavior
- Integration with SambaNova's Llama 3.1 model

## Setup Instructions

1. **Clone the repository**

```bash
git clone <repository-url>
cd pdf-chat-llama
```

2. **Set up a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up SambaNova API credentials**

Create a `.env` file in the project root with the following contents:

```
SAMBANOVA_API_URL=https://api.sambanova.net/llm/v1/generate
SAMBANOVA_API_KEY=your_api_key_here
```

Alternatively, you can set these as environment variables.

5. **Run the Streamlit app**

```bash
streamlit run app.py
```

The application should now be running at http://localhost:8501.

## Usage

1. Use the sidebar to upload a PDF document
2. Click "Process Document" to extract and analyze the content
3. Ask questions about the document in the chat interface
4. View the AI's responses based on the document content

## Requirements

- Python 3.8 or higher
- Streamlit 1.26.0 or higher
- PyMuPDF and PyPDF2 for PDF processing
- sentence-transformers for embedding generation
- SambaNova API access for Llama 3.1 integration

## Cursor Behavior

The app includes custom CSS to ensure appropriate cursor behavior:
- Pointer cursor for buttons and clickable elements
- Text cursor for input fields and text areas
- Copy cursor for file drop zones
- Not-allowed cursor for disabled elements
- Progress cursor for loading states 