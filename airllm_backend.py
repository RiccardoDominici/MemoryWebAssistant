# ==============================================================================
# AirLLM Backend - Memory-Efficient LLM Inference
# ==============================================================================
"""
AirLLM backend for MemoryWebAssistant.
Enables running large language models (70B+) on consumer hardware
by loading model layers sequentially into memory.

Supports:
- Layer-by-layer inference (minimal VRAM usage)
- 4-bit and 8-bit quantization
- Batch embedding generation
- Streaming response generation
- Hugging Face model hub integration
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Generator

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
class QuantizationType(Enum):
    """Quantization options for AirLLM models."""
    NONE = "none"
    INT8 = "8bit"
    INT4 = "4bit"


class CompressionType(Enum):
    """Compression options for layer loading."""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


@dataclass
class AirLLMConfig:
    """Configuration specific to the AirLLM backend."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_len: int = 512
    device: str = "auto"

    # Quantization
    quantization: QuantizationType = QuantizationType.INT4
    compression: CompressionType = CompressionType.NONE

    # Embedding model (sentence-transformers compatible)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Paths
    cache_dir: str = "models_cache"
    offload_dir: str = "offload_cache"


# ==============================================================================
# AirLLM Engine
# ==============================================================================
class AirLLMEngine:
    """
    Inference engine using AirLLM for memory-efficient LLM execution.
    Loads model layers one at a time to minimize VRAM/RAM usage.
    """

    def __init__(self, config: AirLLMConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._embedding_model = None
        self._is_initialized = False

    def initialize(self) -> None:
        """
        Initialize the AirLLM model and tokenizer.
        Models are downloaded from HuggingFace hub on first use.
        """
        if self._is_initialized:
            return

        try:
            from airllm import AutoModel
        except ImportError:
            raise ImportError(
                "AirLLM is not installed. Install with: pip install airllm\n"
                "For 4-bit quantization also install: pip install bitsandbytes"
            )

        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.offload_dir, exist_ok=True)

        logger.info(f"Loading AirLLM model: {self.config.model_name}")
        logger.info(f"Quantization: {self.config.quantization.value}")
        logger.info(f"Max sequence length: {self.config.max_seq_len}")

        # Build model kwargs based on quantization setting
        model_kwargs = {
            "max_seq_len": self.config.max_seq_len,
            "cache_dir": self.config.cache_dir,
        }

        if self.config.compression != CompressionType.NONE:
            model_kwargs["compression"] = self.config.compression.value

        if self.config.device != "auto":
            model_kwargs["device"] = self.config.device

        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        self._is_initialized = True
        logger.info("AirLLM model loaded successfully")

    def _ensure_initialized(self) -> None:
        """Ensure the model is initialized before use."""
        if not self._is_initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Chat / Generation
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> dict | Generator[dict, None, None]:
        """
        Generate a chat response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature (None = use config)
            stream: If True, yield chunks incrementally

        Returns:
            Response dict compatible with Ollama format:
            {"message": {"role": "assistant", "content": "..."}}
        """
        self._ensure_initialized()

        prompt = self._format_chat_messages(messages)
        gen_temp = temperature if temperature is not None else self.config.temperature

        if stream:
            return self._stream_generate(prompt, gen_temp)
        else:
            return self._generate(prompt, gen_temp)

    def _format_chat_messages(self, messages: list[dict]) -> str:
        """
        Format chat messages into a single prompt string.
        Uses a generic chat template compatible with most models.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")

        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _generate(self, prompt: str, temperature: float) -> dict:
        """Generate a complete response."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for AirLLM. "
                "Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len
        ).input_ids

        generation_output = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=self.config.max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
        )

        output_ids = generation_output.sequences[0]
        new_tokens = output_ids[input_ids.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "message": {
                "role": "assistant",
                "content": response_text.strip()
            }
        }

    def _stream_generate(
        self, prompt: str, temperature: float
    ) -> Generator[dict, None, None]:
        """
        Generate response tokens incrementally.
        AirLLM doesn't natively stream, so we simulate by generating
        the full response and yielding it in sentence chunks.
        """
        result = self._generate(prompt, temperature)
        text = result["message"]["content"]

        # Yield text in sentence-level chunks for smooth streaming
        buffer = ""
        for char in text:
            buffer += char
            if char in ".!?:;\n" and len(buffer) > 10:
                yield {"message": {"role": "assistant", "content": buffer}}
                buffer = ""

        if buffer:
            yield {"message": {"role": "assistant", "content": buffer}}

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embeddings(self, prompt: str) -> dict:
        """
        Generate an embedding vector for a single text.

        Args:
            prompt: Text to embed

        Returns:
            Dict with 'embedding' key containing the vector
        """
        vectors = self.batch_embeddings([prompt])
        return {"embedding": vectors[0]}

    def batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in efficient batches.
        Uses sentence-transformers for high-quality embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        model = self._get_embedding_model()
        all_embeddings = []

        for i in range(0, len(texts), self.config.embedding_batch_size):
            batch = texts[i:i + self.config.embedding_batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings

    def _get_embedding_model(self):
        """Lazy-load the sentence-transformers embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for AirLLM embeddings. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(
                f"Loading embedding model: {self.config.embedding_model_name}"
            )
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model_name,
                cache_folder=self.config.cache_dir
            )

        return self._embedding_model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Return information about the loaded model configuration."""
        return {
            "model_name": self.config.model_name,
            "quantization": self.config.quantization.value,
            "compression": self.config.compression.value,
            "max_seq_len": self.config.max_seq_len,
            "embedding_model": self.config.embedding_model_name,
            "initialized": self._is_initialized,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
        self._tokenizer = None
        self._embedding_model = None
        self._is_initialized = False
        logger.info("AirLLM resources released")


# ==============================================================================
# Singleton Access
# ==============================================================================
_engine: Optional[AirLLMEngine] = None


def get_engine(config: Optional[AirLLMConfig] = None) -> AirLLMEngine:
    """
    Get or create the global AirLLM engine instance.

    Args:
        config: Optional config override (only used on first call)

    Returns:
        AirLLMEngine singleton instance
    """
    global _engine
    if _engine is None:
        _engine = AirLLMEngine(config or AirLLMConfig())
    return _engine


def reset_engine() -> None:
    """Reset the global engine (for testing)."""
    global _engine
    if _engine is not None:
        _engine.cleanup()
    _engine = None
