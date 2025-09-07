"""FastAPI server for AI Ad Generation System."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import get_settings, validate_environment
from exceptions import (
    AIAdGenerationError, PDFProcessingError, 
    VectorStoreError, ImageGenerationError, AgentError
)
from langgraph_agent import AdGenerationAgent
from models import (
    AdGenerationRequest, AdGenerationResponse, UploadResponse,
    SearchResponse, SearchResult, HealthResponse
)
from pdf_vectorizer import PDFVectorizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
vectorizer: Optional[PDFVectorizer] = None
agent: Optional[AdGenerationAgent] = None
settings = get_settings()


async def initialize_components():
    """Initialize application components."""
    global vectorizer, agent
    
    try:
        logger.info("Initializing application components...")
        
        # Validate environment
        env_valid, env_errors = validate_environment()
        if not env_valid:
            for error in env_errors:
                logger.error(error)
            raise Exception(f"Environment validation failed: {'; '.join(env_errors)}")
        
        # Initialize vectorizer
        logger.info("Initializing PDF vectorizer...")
        vectorizer = PDFVectorizer(settings)
        
        # Initialize agent
        logger.info("Initializing ad generation agent...")
        agent = AdGenerationAgent(settings, vectorizer)
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return False


async def cleanup_components():
    """Cleanup application components."""
    logger.info("Cleaning up application components...")
    # Add any cleanup logic here if needed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    success = await initialize_components()
    if not success:
        logger.error("Failed to initialize components")
    
    yield
    
    # Shutdown
    await cleanup_components()


# Create FastAPI application
app = FastAPI(
    title="AI Ad Generation System",
    description="Professional AI-powered advertisement generation system using RAG and FLUX",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(PDFProcessingError)
async def pdf_processing_exception_handler(request, exc):
    logger.error(f"PDF processing error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"PDF processing failed: {str(exc)}"}
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request, exc):
    logger.error(f"Vector store error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Vector store operation failed: {str(exc)}"}
    )


@app.exception_handler(ImageGenerationError)
async def image_generation_exception_handler(request, exc):
    logger.error(f"Image generation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Image generation failed: {str(exc)}"}
    )


@app.exception_handler(AgentError)
async def agent_exception_handler(request, exc):
    logger.error(f"Agent error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Agent operation failed: {str(exc)}"}
    )


# API Routes

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Ad Generation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    errors = []
    
    # Check component initialization
    vectorizer_ready = vectorizer is not None
    agent_ready = agent is not None
    
    # Check API keys
    gemini_key_set = bool(settings.gemini_api_key)
    hf_key_set = bool(settings.huggingface_api_key)
    
    if not gemini_key_set:
        errors.append("GEMINI_API_KEY not configured")
    if not hf_key_set:
        errors.append("HUGGINGFACE_API_KEY not configured")
    
    # Check vector store
    vector_store_exists = False
    document_count = 0
    
    if vectorizer_ready:
        try:
            stats = vectorizer.get_collection_stats()
            vector_store_exists = stats.get("exists", False)
            document_count = stats.get("count", 0)
        except Exception as e:
            errors.append(f"Vector store check failed: {e}")
    
    # Check agent status
    if agent_ready:
        try:
            agent_status = agent.get_agent_status()
            if not agent_status.get("gemini_available", False):
                errors.append("Gemini API not accessible")
            if not agent_status.get("image_generator_available", False):
                errors.append("Image generator not available")
        except Exception as e:
            errors.append(f"Agent status check failed: {e}")
    
    return HealthResponse(
        api_status="healthy" if not errors else "degraded",
        vectorizer_initialized=vectorizer_ready,
        agent_initialized=agent_ready,
        gemini_api_key_configured=gemini_key_set,
        huggingface_api_key_configured=hf_key_set,
        vector_store_exists=vector_store_exists,
        document_count=document_count,
        errors=errors
    )


@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not vectorizer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vectorizer not initialized"
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    # Check file size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Save file
    file_path = Path(settings.upload_dir) / file.filename
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )
    
    # Process PDF
    try:
        logger.info(f"Processing uploaded PDF: {file.filename}")
        documents = vectorizer.load_pdf(str(file_path))
        
        # Check if vector store exists
        existing_store = vectorizer.load_existing_vector_store()
        
        if existing_store:
            success = vectorizer.add_pdf_to_existing_store(str(file_path))
        else:
            success = vectorizer.create_vector_store(documents)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document vectors"
            )
        
        logger.info(f"Successfully processed {file.filename} with {len(documents)} chunks")
        
        return UploadResponse(
            success=True,
            message=f"Successfully uploaded and processed {file.filename}",
            filename=file.filename,
            processed_chunks=len(documents),
            file_size_mb=round(file_size_mb, 2)
        )
        
    except (PDFProcessingError, VectorStoreError):
        # Let the exception handlers deal with these
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during PDF processing: {str(e)}"
        )
    finally:
        # Clean up uploaded file if processing failed
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up uploaded file: {e}")


@app.post("/generate-ad/", response_model=AdGenerationResponse)
async def generate_advertisement(request: AdGenerationRequest):
    """Generate an advertisement based on user request."""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ad generation agent not initialized"
        )
    
    logger.info(f"Generating ad for request: {request.user_request}")
    
    try:
        result = agent.process_request(request.user_request)
        
        return AdGenerationResponse(
            success=result["success"],
            user_request=result["user_request"],
            ad_prompt=result["ad_prompt"],
            generated_image=result["generated_image"],
            error_message=result["error_message"],
            context_used=result.get("context_used"),
            messages=result["messages"]
        )
        
    except (AgentError, ImageGenerationError):
        # Let the exception handlers deal with these
        raise
    except Exception as e:
        logger.error(f"Unexpected error during ad generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during ad generation: {str(e)}"
        )


@app.get("/search/", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return")
):
    """Search for similar documents in the vector store."""
    if not vectorizer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vectorizer not initialized"
        )
    
    logger.info(f"Searching documents with query: {query}")
    
    try:
        results = vectorizer.search_similar(query, k=k)
        
        search_results = [
            SearchResult(
                content=doc.page_content,
                source=doc.metadata.get("file_name", "Unknown"),
                score=None  # Chroma doesn't return scores by default
            )
            for doc in results
        ]
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except VectorStoreError:
        # Let the exception handler deal with this
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during search: {str(e)}"
        )


@app.get("/stats/", response_model=dict)
async def get_statistics():
    """Get system statistics."""
    stats = {
        "system_status": "operational",
        "components": {
            "vectorizer": vectorizer is not None,
            "agent": agent is not None
        },
        "configuration": {
            "upload_directory": settings.upload_dir,
            "vector_store_path": settings.vector_store_path,
            "max_file_size_mb": settings.max_file_size_mb
        }
    }
    
    # Add vector store statistics
    if vectorizer:
        try:
            vector_stats = vectorizer.get_collection_stats()
            stats["vector_store"] = vector_stats
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {e}")
            stats["vector_store"] = {"error": str(e)}
    
    # Add agent statistics
    if agent:
        try:
            agent_stats = agent.get_agent_status()
            stats["agent"] = agent_stats
        except Exception as e:
            logger.warning(f"Failed to get agent stats: {e}")
            stats["agent"] = {"error": str(e)}
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastapi_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )