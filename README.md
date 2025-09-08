# AI Ad Generation System @ beyond apps group

A professional AI-powered advertisement generation system that combines Retrieval-Augmented Generation (RAG) with FLUX image generation to create compelling visual advertisements from text descriptions.

## Features

- **RAG-Powered Context**: Upload PDF documents to build a knowledge base for contextual ad generation
- **Multi-Model AI**: Uses Google Gemini (with Phi-3-mini fallback) for intelligent prompt generation
- **FLUX Image Generation**: Generates high-quality advertisement images using FLUX.1-schnell model
- **FastAPI Backend**: RESTful API with comprehensive error handling and validation
- **Streamlit Frontend**: User-friendly web interface for document upload and ad generation
- **Vector Search**: ChromaDB-powered similarity search through uploaded documents
- **Health Monitoring**: Built-in system health checks and performance monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │────│   FastAPI       │────│   AI Components │
│  (Frontend)     │    │   (Backend)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        ├─ Gemini API
                                │                        ├─ FLUX Generator
                                │                        └─ Local LLM
                                │
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (ChromaDB)    │
                       └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- API Keys:
  - Google Gemini API key (required)
  - Hugging Face API key (required)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai-ad-generation-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
Create a `.env` file in the project root:
```bash
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional Configuration
VECTOR_STORE_PATH=./chroma_db
UPLOAD_DIR=./uploaded_pdfs
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

4. **Get API Keys:**
   - **Gemini API**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Hugging Face**: Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens)

## Usage

### Starting the Application

1. **Start the FastAPI server:**
```bash
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

2. **Launch the Streamlit interface:**
```bash
streamlit run streamlit_ui.py
```

3. **Access the application:**
   - Web UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Using the System

1. **Upload Documents** (Optional):
   - Upload PDF files containing advertising guidelines, brand information, or design principles
   - Documents are processed and stored in the vector database for contextual generation

2. **Generate Advertisements**:
   - Describe your advertisement requirements
   - The system will:
     - Retrieve relevant context from uploaded documents
     - Generate an optimized prompt using Gemini AI
     - Create a visual advertisement using FLUX

3. **Search Knowledge Base**:
   - Search through uploaded documents
   - Find relevant information for manual reference

## API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Comprehensive health check
- `POST /upload-pdf/` - Upload and process PDF documents
- `POST /generate-ad/` - Generate advertisement from text description
- `GET /search/` - Search uploaded documents
- `GET /stats/` - System statistics and performance metrics

### Example API Usage

```python
import requests

# Generate an advertisement
response = requests.post("http://localhost:8000/generate-ad/", 
    json={
        "user_request": "Create a modern smartphone ad targeting tech-savvy millennials"
    }
)

result = response.json()
if result["success"]:
    # result["generated_image"] contains base64-encoded PNG 
    # result["ad_prompt"] contains the generated prompt
    print(f"Generated prompt: {result['ad_prompt']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `HUGGINGFACE_API_KEY` | Hugging Face API key | Required |
| `VECTOR_STORE_PATH` | ChromaDB storage path | `./chroma_db` |
| `UPLOAD_DIR` | PDF upload directory | `./uploaded_pdfs` |
| `MAX_FILE_SIZE_MB` | Maximum PDF file size | `50` |
| `CHUNK_SIZE` | Text chunking size | `500` |
| `CHUNK_OVERLAP` | Text chunk overlap | `50` |
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `GEMINI_MODEL` | Gemini model version | `gemini-1.5-flash` |
| `FLUX_MODEL` | FLUX model name | `black-forest-labs/FLUX.1-schnell` |

### Model Configuration

The system supports configuration for:
- Image dimensions (default: 1024x1024)
- FLUX generation steps
- Guidance scale
- Text chunking parameters

## Standalone Testing

For testing without the full FastAPI server, use the standalone test script:

```bash
python test.py
```

This script:
- Tests prompt generation capabilities
- Runs a comprehensive test suite with various categories
- Generates detailed reports and examples
- Works independently of the main application

## Project Structure

```
ai-ad-generation-system/
├── config.py              # Configuration management
├── exceptions.py          # Custom exception classes
├── fastapi_server.py      # FastAPI application server
├── flux_generator.py      # FLUX image generation
├── langgraph_agent.py     # Main AI agent logic
├── local_llm.py          # Local LLM fallback (Phi-3-mini)
├── models.py             # Pydantic data models
├── pdf_vectorizer.py     # PDF processing and vectorization
├── streamlit_ui.py       # Streamlit web interface
├── test.py               # Standalone testing script
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Components

### PDF Vectorizer (`pdf_vectorizer.py`)
- Processes PDF documents using PyPDFLoader
- Splits text into chunks with configurable size and overlap
- Stores vectors in ChromaDB with HuggingFace embeddings
- Provides similarity search functionality

### AI Agent (`langgraph_agent.py`)
- Orchestrates the complete ad generation workflow
- Integrates RAG context retrieval with prompt generation
- Falls back to Phi-3-mini if Gemini API is unavailable
- Handles end-to-end request processing

### Image Generator (`flux_generator.py`)
- Interfaces with Hugging Face Inference API
- Uses FLUX.1-schnell model for high-quality image generation
- Returns base64-encoded PNG images
- Includes prompt validation and safety checks

### Web Interface (`streamlit_ui.py`)
- Full-featured web application
- Document upload and management
- Real-time ad generation with progress tracking
- System monitoring and health checks
- Download functionality for generated images

## Error Handling

The system includes comprehensive error handling:

- **PDF Processing**: Invalid files, extraction failures, size limits
- **API Errors**: Rate limits, authentication failures, network issues
- **Vector Store**: Initialization failures, search errors
- **Image Generation**: Invalid prompts, generation failures
- **Validation**: Input validation, type checking

## Performance

- **Typical Generation Time**: 5-15 seconds per advertisement
- **Supported File Size**: Up to 50MB PDFs
- **Concurrent Requests**: Handled via FastAPI async support
- **Vector Search**: Sub-second response times for typical queries

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Verify API keys are set in `.env` file
   - Check API key permissions and quotas

2. **Memory Issues**:
   - Reduce chunk size if processing large documents
   - Monitor system memory usage

3. **Generation Failures**:
   - Check Hugging Face API status
   - Verify network connectivity
   - Review prompt validation errors

4. **Vector Store Issues**:
   - Ensure write permissions for storage directory
   - Check disk space availability

### Logs and Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check application logs for detailed error information and performance metrics.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Test with the standalone script
4. Submit an issue with detailed information

## Acknowledgments

- Google Gemini for advanced language understanding
- Hugging Face for FLUX image generation
- LangChain for document processing
- ChromaDB for vector storage
- FastAPI and Streamlit for the application framework
