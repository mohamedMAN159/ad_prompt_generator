"""Pydantic models for API request/response validation."""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator


class AdGenerationRequest(BaseModel):
    """Request model for ad generation."""
    
    user_request: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Description of the advertisement to generate"
    )
    
    @validator('user_request')
    def validate_user_request(cls, v):
        if not v or not v.strip():
            raise ValueError('User request cannot be empty')
        return v.strip()


class Message(BaseModel):
    """Message model for process feedback."""
    
    type: str = Field(..., description="Message type (info, success, warning, error)")
    content: str = Field(..., description="Message content")


class AdGenerationResponse(BaseModel):
    """Response model for ad generation."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    user_request: str = Field(..., description="Original user request")
    ad_prompt: str = Field(..., description="Generated prompt for image creation")
    generated_image: Optional[str] = Field(None, description="Base64 encoded image")
    error_message: str = Field("", description="Error message if operation failed")
    context_used: Optional[bool] = Field(None, description="Whether context from documents was used")
    messages: List[Message] = Field(default_factory=list, description="Process messages")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    filename: Optional[str] = Field(None, description="Name of uploaded file")
    processed_chunks: Optional[int] = Field(None, description="Number of processed document chunks")
    file_size_mb: Optional[float] = Field(None, description="File size in MB")


class SearchResult(BaseModel):
    """Model for search result item."""
    
    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, description="Source file name")
    score: Optional[float] = Field(None, description="Similarity score")


class SearchResponse(BaseModel):
    """Response model for document search."""
    
    query: str = Field(..., description="Search query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    api_status: str = Field(..., description="API status")
    vectorizer_initialized: bool = Field(..., description="Whether vectorizer is ready")
    agent_initialized: bool = Field(..., description="Whether agent is ready")
    gemini_api_key_configured: bool = Field(..., description="Whether Gemini API key is set")
    huggingface_api_key_configured: bool = Field(..., description="Whether HuggingFace API key is set")
    vector_store_exists: bool = Field(..., description="Whether vector store exists")
    document_count: int = Field(0, description="Number of documents in vector store")
    errors: List[str] = Field(default_factory=list, description="Any configuration errors")