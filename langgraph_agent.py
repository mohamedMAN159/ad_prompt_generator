# langgraph_agent.py
"""
AI agent for ad generation using Gemini (with Phi-3-mini fallback) and RAG.
"""

import logging
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai import GenerativeModel

from config import Settings
from exceptions import AgentError, ImageGenerationError
from flux_generator import FluxImageGenerator
from pdf_vectorizer import PDFVectorizer
from local_llm import LocalLLM  # ✅ Import fallback model

logger = logging.getLogger(__name__)


class AdGenerationAgent:
    """AI agent that combines RAG with Gemini or local Phi-3-mini."""

    def __init__(self, settings: Settings, vectorizer: PDFVectorizer):
        self.settings = settings
        self.vectorizer = vectorizer
        self.use_gemini = False  # Track active backend

        # Try to initialize Gemini
        if not settings.gemini_api_key:
            logger.warning("No GEMINI_API_KEY → using Phi-3-mini (local)")
            self.model = LocalLLM()
            self.use_gemini = False
        else:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.model = GenerativeModel(settings.gemini_model)
                self.use_gemini = True
                logger.info("✅ Gemini model initialized")
            except Exception as e:
                logger.warning(f"Gemini failed → using Phi-3-mini: {e}")
                self.model = LocalLLM()
                self.use_gemini = False

        # Initialize image generator
        try:
            self.image_generator = FluxImageGenerator(settings)
            logger.info("✅ Image generator ready")
        except Exception as e:
            logger.error(f"❌ Image generator init failed: {e}")
            raise AgentError(f"Image generator failed: {e}")

    def retrieve_context(self, user_request: str, k: int = 5) -> str:
        """Retrieve relevant context from vector store."""
        try:
            docs = self.vectorizer.search_similar(user_request, k=k)
            if not docs:
                return self._get_default_context()
            return "\n\n".join([
                f"From {doc.metadata.get('file_name', 'unknown')}: {doc.page_content.strip()}"
                for doc in docs
            ])
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return self._get_default_context()

    def _get_default_context(self) -> str:
        return """
        Professional advertising principles:
        - Clean layout with visual hierarchy
        - Strong call-to-action
        - Brand-consistent colors and fonts
        - Audience-resonant imagery
        - Mobile-friendly design
        """

    def generate_ad_prompt(self, user_request: str, context: str) -> str:
        """Generate prompt using Gemini or Phi-3-mini."""
        system_prompt = """
        You are an expert ad creative director.
        Create a detailed, concise image generation prompt (<500 chars) that includes:
        - Visual style, composition
        - Color palette, mood
        - Typography, branding
        - Target audience
        """

        full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser Request: {user_request}"

        try:
            if self.use_gemini:
                logger.info("🧠 Generating prompt with Gemini")
                response = self.model.generate_content(full_prompt)
                return response.text.strip() if response.text else ""
            else:
                logger.info("🧠 Generating prompt with Phi-3-mini (local)")
                return self.model.generate(full_prompt)

        except Exception as e:
            logger.warning(f"Prompt generation failed: {e}")

            # If Gemini quota exceeded, switch to Phi
            if self.use_gemini and "429" in str(e):
                logger.info("🔁 Switching to Phi-3-mini due to quota")
                phi = LocalLLM()
                return phi.generate(full_prompt)

            # Fallback
            return f"Professional ad for {user_request}, clean modern style"

    def process_request(self, user_request: str) -> Dict[str, Any]:
        """Process user request end-to-end."""
        if not user_request or not user_request.strip():
            return {
                "success": False,
                "error_message": "Request cannot be empty",
                "user_request": user_request,
                "ad_prompt": "",
                "generated_image": None,
                "messages": []
            }

        messages = []
        try:
            # Step 1: Retrieve context
            context = self.retrieve_context(user_request)
            messages.append({
                "type": "info",
                "content": "Retrieved context from knowledge base"
            })

            # Step 2: Generate prompt
            ad_prompt = self.generate_ad_prompt(user_request, context)
            messages.append({
                "type": "info",
                "content": f"Generated prompt using {'Gemini' if self.use_gemini else 'Phi-3-mini'}"
            })

            # Step 3: Validate prompt
            is_valid, error = self.image_generator.validate_prompt(ad_prompt)
            if not is_valid:
                return {
                    "success": False,
                    "error_message": error,
                    "messages": messages
                }

            # Step 4: Generate image
            generated_image = self.image_generator.generate_image(ad_prompt)
            messages.append({
                "type": "success",
                "content": "Image generated with FLUX AI"
            })

            return {
                "success": True,
                "user_request": user_request,
                "ad_prompt": ad_prompt,
                "generated_image": generated_image,
                "error_message": "",
                "messages": messages
            }

        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            messages.append({
                "type": "error",
                "content": str(e)
            })
            return {
                "success": False,
                "error_message": str(e),
                "messages": messages
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        gemini_ok = False
        if self.use_gemini:
            try:
                test = self.model.generate_content("Hello")
                gemini_ok = bool(test.text)
            except:
                gemini_ok = False

        vector_stats = self.vectorizer.get_collection_stats()
        return {
            "gemini_available": gemini_ok,
            "active_backend": "Gemini" if self.use_gemini else "Phi-3-mini (local)",
            "image_generator_available": self.image_generator is not None,
            "vector_store_exists": vector_stats.get("exists", False),
            "document_count": vector_stats.get("count", 0),
            "flux_model": self.settings.flux_model

        }
