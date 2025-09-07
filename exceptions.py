"""Custom exceptions for the AI Ad Generation System."""


class AIAdGenerationError(Exception):
    """Base exception for AI Ad Generation System."""
    pass


class VectorizerError(AIAdGenerationError):
    """Exception raised for vectorizer-related errors."""
    pass


class PDFProcessingError(VectorizerError):
    """Exception raised when PDF processing fails."""
    pass


class VectorStoreError(VectorizerError):
    """Exception raised for vector store operations."""
    pass


class ImageGenerationError(AIAdGenerationError):
    """Exception raised when image generation fails."""
    pass


class AgentError(AIAdGenerationError):
    """Exception raised for agent-related errors."""
    pass


class ConfigurationError(AIAdGenerationError):
    """Exception raised for configuration-related errors."""
    pass

# exceptions.py
class AIAdGenerationError(Exception):
    """Base exception for AI Ad Generation System."""
    pass


class ImageGenerationError(AIAdGenerationError):
    """Exception raised when image generation fails."""
    pass