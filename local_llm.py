# local_llm.py
"""
Wrapper around llama.cpp to run Phi-3-mini locally.
Used as fallback when Gemini is unavailable.
"""

from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Wrapper around llama.cpp for running Phi-3-mini locally.
    """

    def __init__(self, model_path: str = "./models/phi-3-mini-4k-instruct-q4.gguf"):
        """
        Initialize local Phi-3-mini model.

        Args:
            model_path: Path to the quantized GGUF model file
        """
        logger.info(f"Loading local LLM model: {model_path}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,           # Context length (Phi-3 supports up to 4k)
                n_threads=8,          # CPU threads (adjust based on your CPU)
                n_gpu_layers=20,      # Offload layers to GPU (set 0 for CPU-only)
                verbose=False         # Reduce verbose logs
            )
            logger.info("✅ Local LLM (Phi-3-mini) loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load local LLM: {e}")
            raise RuntimeError(f"Could not load model from {model_path}. Did you download it?")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Generate text response from Phi-3-mini.

        Args:
            prompt: Input text prompt
            max_tokens: Max tokens to generate

        Returns:
            Generated string
        """
        try:
            logger.info(f"Generating with local LLM: {len(prompt)}-char prompt")
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "User:", "Assistant:", "#"],
                echo=False
            )
            text = response["choices"][0]["text"].strip()
            logger.info("✅ Local LLM generation successful")
            return text
        except Exception as e:
            logger.error(f"❌ Local LLM generation failed: {e}")
            return f"Professional ad for {prompt.split('User Request:')[-1].strip()}"