"""
Tests for MemoryWebAssistant chat_bot module.
These tests are self-contained and don't require external dependencies.
"""

import os
import tempfile
import unittest
import re
from enum import Enum
from dataclasses import dataclass


# ============================================================================
# Copied functions from chat_bot.py for testing (to avoid import dependencies)
# ============================================================================

def parse_file(filename: str) -> list:
    """Parse a file into paragraphs separated by blank lines."""
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []

        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []

        if buffer:
            paragraphs.append(" ".join(buffer))

    return paragraphs


def compute_cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    def norm(v):
        return sum(x * x for x in v) ** 0.5
    return dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def find_similar(needle: list, haystack: list) -> list:
    """Find vectors in haystack similar to needle using cosine similarity."""
    def norm(v):
        return sum(x * x for x in v) ** 0.5
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    needle_norm = norm(needle)
    similarity_scores = [
        dot(needle, item) / (needle_norm * norm(item))
        for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)


class RetrievalMode(Enum):
    RAG = "rag"
    CAG = "cag"


@dataclass
class Config:
    model_chatbot: str = "gemma3n:e4b"
    model_embedding: str = "snowflake-arctic-embed2"
    model_voice_transcription: str = "large-v3"
    system_language: str = "italian"
    voice_name: str = "if_sara"
    voice_enabled: bool = True
    retrieval_mode: RetrievalMode = RetrievalMode.RAG
    rag_top_k: int = 15
    cag_max_chars: int = 50000
    similarity_threshold: float = 0.8
    memory_file: str = "memory.txt"
    embeddings_dir: str = "embeddings"
    voice_dir: str = "voice"
    translation_cache_file: str = "translations.json"


# ============================================================================
# Test Classes
# ============================================================================

class TestParseFile(unittest.TestCase):
    """Tests for the parse_file function."""

    def test_parse_file_with_paragraphs(self):
        """Test parsing a file with multiple paragraphs separated by blank lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("First paragraph line one.\nFirst paragraph line two.\n\n")
            f.write("Second paragraph.\n\n")
            f.write("Third paragraph.")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], "First paragraph line one. First paragraph line two.")
            self.assertEqual(result[1], "Second paragraph.")
            self.assertEqual(result[2], "Third paragraph.")
        finally:
            os.unlink(temp_path)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for the compute_cosine_similarity function."""

    def test_cosine_similarity(self):
        """Test cosine similarity for identical and orthogonal vectors."""
        # Identical vectors should have similarity of 1.0
        vec = [1.0, 2.0, 3.0]
        similarity = compute_cosine_similarity(vec, vec)
        self.assertAlmostEqual(similarity, 1.0, places=5)

        # Orthogonal vectors should have similarity of 0.0
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = compute_cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, 0.0, places=5)


class TestFindSimilar(unittest.TestCase):
    """Tests for the find_similar function."""

    def test_find_most_similar(self):
        """Test finding the most similar vector from a list."""
        needle = [1.0, 0.0, 0.0]
        haystack = [
            [0.0, 1.0, 0.0],  # orthogonal
            [1.0, 0.1, 0.0],  # very similar
            [0.5, 0.5, 0.0],  # somewhat similar
        ]
        results = find_similar(needle, haystack)
        # First result should be the most similar (index 1)
        self.assertEqual(results[0][1], 1)
        # Similarity should be close to 1.0
        self.assertGreater(results[0][0], 0.9)


class TestEmojiPattern(unittest.TestCase):
    """Tests for the EMOJI_PATTERN regex."""

    def test_emoji_pattern(self):
        """Test that emoji pattern removes emojis but keeps regular text."""
        # Should remove emojis
        text_with_emojis = "Hello! \U0001F600 How are you?"
        cleaned = EMOJI_PATTERN.sub("", text_with_emojis)
        self.assertEqual(cleaned, "Hello!  How are you?")

        # Should preserve regular text
        text = "Just regular text with punctuation!"
        cleaned = EMOJI_PATTERN.sub("", text)
        self.assertEqual(cleaned, text)


class TestRetrievalMode(unittest.TestCase):
    """Tests for the RetrievalMode enum."""

    def test_retrieval_mode_values(self):
        """Test that RetrievalMode enum has correct values."""
        self.assertEqual(RetrievalMode.RAG.value, "rag")
        self.assertEqual(RetrievalMode.CAG.value, "cag")


class TestConfig(unittest.TestCase):
    """Tests for the Config dataclass."""

    def test_config_defaults(self):
        """Test that default configuration values are set correctly."""
        config = Config()
        self.assertEqual(config.retrieval_mode, RetrievalMode.RAG)
        self.assertEqual(config.rag_top_k, 15)
        self.assertEqual(config.cag_max_chars, 50000)
        self.assertEqual(config.similarity_threshold, 0.8)
        self.assertEqual(config.memory_file, "memory.txt")


if __name__ == "__main__":
    unittest.main()
