"""PDF processing and vectorization module."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings
from exceptions import PDFProcessingError, VectorStoreError


logger = logging.getLogger(__name__)


class PDFVectorizer:
    """Handles PDF processing and vector storage operations."""
    
    def __init__(self, settings: Settings):
        """Initialize the PDF vectorizer.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.collection_name = settings.collection_name
        self.persist_directory = settings.vector_store_path
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise VectorStoreError(f"Failed to initialize embeddings: {e}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process a PDF file into document chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks
            
        Raises:
            PDFProcessingError: If PDF loading or processing fails
        """
        if not Path(file_path).exists():
            raise PDFProcessingError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise PDFProcessingError(f"Invalid file type: {file_path}")
        
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                raise PDFProcessingError(f"No content extracted from PDF: {file_path}")
            
            logger.info(f"Splitting {len(documents)} pages into chunks")
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            file_name = Path(file_path).name
            for chunk in chunks:
                chunk.metadata["file_name"] = file_name
                chunk.metadata["file_path"] = file_path
            
            logger.info(f"Created {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise PDFProcessingError(f"Failed to process PDF: {e}")
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create a new vector store with the provided documents.
        
        Args:
            documents: List of documents to vectorize
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for vector store creation")
            return False
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise VectorStoreError(f"Failed to create vector store: {e}")
    
    def add_pdf_to_existing_store(self, file_path: str) -> bool:
        """Add a PDF to an existing vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If adding to vector store fails
        """
        try:
            vector_store = self.load_existing_vector_store()
            if not vector_store:
                logger.error("No existing vector store found")
                return False
            
            documents = self.load_pdf(file_path)
            logger.info(f"Adding {len(documents)} chunks to existing vector store")
            
            vector_store.add_documents(documents)
            logger.info("Documents added successfully")
            return True
            
        except PDFProcessingError:
            raise
        except Exception as e:
            logger.error(f"Failed to add PDF to vector store: {e}")
            raise VectorStoreError(f"Failed to add PDF to vector store: {e}")
    
    def load_existing_vector_store(self) -> Optional[Chroma]:
        """Load an existing vector store if it exists.
        
        Returns:
            Chroma vector store instance or None if doesn't exist
        """
        if not Path(self.persist_directory).exists():
            logger.info("Vector store directory doesn't exist")
            return None
        
        try:
            logger.info("Loading existing vector store")
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            return vector_store
            
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            return None
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            VectorStoreError: If search fails
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            vector_store = self.load_existing_vector_store()
            if not vector_store:
                logger.warning("No vector store available for search")
                return []
            
            logger.info(f"Searching for: '{query}' (k={k})")
            results = vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            vector_store = self.load_existing_vector_store()
            if not vector_store:
                return {"exists": False, "count": 0}
            
            # Try to get collection info
            collection = vector_store._collection
            count = collection.count()
            
            return {
                "exists": True,
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            return {"exists": False, "count": 0, "error": str(e)}