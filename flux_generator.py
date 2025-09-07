"""
flux_generator.py

Image generation module using the FLUX.1-schnell model hosted on Hugging Face.
Generates images from text prompts using the Hugging Face Inference API.
Returns images as base64-encoded PNG strings for web display.
"""

import base64
import logging
from io import BytesIO
from typing import Optional

from PIL import Image
from huggingface_hub import InferenceClient
from config import Settings
from exceptions import ImageGenerationError

logger = logging.getLogger(__name__)


class FluxImageGenerator:
    """
    Handles image generation using the FLUX.1-schnell model via Hugging Face API.
    Does not require local model download.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the FLUX image generator.

        Args:
            settings: Application settings containing HUGGINGFACE_API_KEY

        Raises:
            ImageGenerationError: If API key is missing or client fails to initialize
        """
        self.settings = settings

        if not settings.huggingface_api_key:
            raise ImageGenerationError("HUGGINGFACE_API_KEY is required in environment variables")

        try:
            self.client = InferenceClient(token=settings.huggingface_api_key)
            logger.info("✅ FLUX image generator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hugging Face client: {e}")
            raise ImageGenerationError(f"Failed to initialize image generator: {e}")

    def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> Optional[str]:
        """
        Generate an image using FLUX.1-schnell.

        Args:
            prompt: Text description for the image
            width: Image width (defaults to config)
            height: Image height (defaults to config)
            num_inference_steps: Number of inference steps (may be ignored by hosted API)
            guidance_scale: Guidance scale for generation (may be ignored by hosted API)

        Returns:
            Base64-encoded PNG string, or None if generation fails
        """
        if not prompt or not prompt.strip():
            raise ImageGenerationError("Prompt cannot be empty")

        width = width or self.settings.default_image_width
        height = height or self.settings.default_image_height

        # Ensure numeric values
        try:
            num_inference_steps = int(num_inference_steps or self.settings.flux_steps)
        except Exception:
            num_inference_steps = self.settings.flux_steps

        try:
            guidance_scale = float(
                guidance_scale if guidance_scale is not None else self.settings.flux_guidance_scale
            )
        except Exception:
            guidance_scale = 0.0

        logger.info(f"🎨 Generating image: '{prompt[:80]}...'")
        logger.info(f"⚙️ Parameters: {width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}")

        try:
            # Build kwargs dynamically depending on API support
            kwargs = {
                "model": self.settings.flux_model,
                "prompt": prompt,
                "width": width,
                "height": height,
            }

            # Only include optional params if not using hosted inference
            if "flux.1-schnell" not in self.settings.flux_model.lower():
                kwargs["num_inference_steps"] = num_inference_steps
                kwargs["guidance_scale"] = guidance_scale
                logger.info("Using local/diffusers-compatible parameters")
            else:
                logger.info("Using Hugging Face hosted Inference API (ignoring steps & guidance_scale)")

            logger.debug(f"[DEBUG] Final request kwargs: {kwargs}")

            image_bytes = self.client.text_to_image(**kwargs)

            # ✅ Handle different return types from Hugging Face API
            if isinstance(image_bytes, Image.Image):
                image = image_bytes  # already a PIL image
            elif hasattr(image_bytes, "read"):
                image = Image.open(image_bytes)  # file-like
            else:
                image = Image.open(BytesIO(image_bytes))  # raw bytes

            if image.size[0] <= 0 or image.size[1] <= 0:
                raise ImageGenerationError("Generated image has invalid dimensions")

            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.info(f"✅ Image generated successfully: {image.size[0]}x{image.size[1]}")
            return base64_image

        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            raise ImageGenerationError(f"Image generation failed: {e}")
    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Validate image generation prompt.

        Args:
            prompt: The prompt to validate

        Returns:
            (is_valid, error_message)
        """
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"

        if len(prompt) > 1000:
            return False, "Prompt too long (max 1000 characters)"

        blocked_terms = ["nsfw", "explicit", "nude", "sexual"]
        prompt_lower = prompt.lower()

        for term in blocked_terms:
            if term in prompt_lower:
                return False, f"Prompt contains blocked content: {term}"

        return True, ""


