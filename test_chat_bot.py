"""
Tests for MemoryWebAssistant chat_bot module.
These tests are self-contained and don't require external dependencies.
"""

import os
import json
import tempfile
import unittest
import re
import hashlib
from enum import Enum
from dataclasses import dataclass
from typing import Optional


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


def save_embeddings(filename: str, embeddings: dict, embeddings_dir: str) -> None:
    """Save embeddings to a JSON file in the embeddings directory."""
    name = filename.split(".")[0]
    os.makedirs(embeddings_dir, exist_ok=True)
    with open(f"{embeddings_dir}/{name}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename: str, embeddings_dir: str) -> Optional[dict]:
    """Load embeddings from a JSON file if it exists."""
    path = f"{embeddings_dir}/{filename}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_translation_cache(cache_file: str) -> dict:
    """Load the translation cache from disk."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_translation_cache(cache: dict, cache_file: str) -> None:
    """Save the translation cache to disk."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


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

    def test_parse_empty_file(self):
        """Test parsing an empty file returns empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_path)

    def test_parse_single_line(self):
        """Test parsing a file with a single line."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Single line of text.")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "Single line of text.")
        finally:
            os.unlink(temp_path)

    def test_parse_only_blank_lines(self):
        """Test parsing a file with only blank lines returns empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("\n\n\n\n")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_path)

    def test_parse_multiple_blank_lines_between_paragraphs(self):
        """Test that multiple blank lines still separate paragraphs correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("First paragraph.\n\n\n\nSecond paragraph.")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "First paragraph.")
            self.assertEqual(result[1], "Second paragraph.")
        finally:
            os.unlink(temp_path)

    def test_parse_file_with_leading_whitespace(self):
        """Test parsing handles leading/trailing whitespace in lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("   Leading spaces.\n   More spaces.   \n\nNext paragraph.")
            temp_path = f.name

        try:
            result = parse_file(temp_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "Leading spaces. More spaces.")
            self.assertEqual(result[1], "Next paragraph.")
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

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        similarity = compute_cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, -1.0, places=5)

    def test_different_magnitudes_same_direction(self):
        """Test that vectors with same direction but different magnitudes have similarity 1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [2.0, 4.0, 6.0]  # Same direction, double magnitude
        similarity = compute_cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_partial_similarity(self):
        """Test vectors with partial similarity."""
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 1.0]
        similarity = compute_cosine_similarity(vec_a, vec_b)
        # Expected: 1 / (1 * sqrt(2)) = 0.7071...
        self.assertAlmostEqual(similarity, 0.7071, places=3)

    def test_high_dimensional_vectors(self):
        """Test cosine similarity with high-dimensional vectors."""
        vec_a = [1.0] * 100
        vec_b = [1.0] * 100
        similarity = compute_cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, 1.0, places=5)


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

    def test_find_similar_returns_sorted(self):
        """Test that results are sorted by similarity in descending order."""
        needle = [1.0, 0.0]
        haystack = [
            [0.5, 0.5],   # medium similarity
            [1.0, 0.0],   # exact match
            [0.0, 1.0],   # orthogonal
        ]
        results = find_similar(needle, haystack)
        # Check scores are in descending order
        scores = [r[0] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # First should be exact match (index 1)
        self.assertEqual(results[0][1], 1)
        # Last should be orthogonal (index 2)
        self.assertEqual(results[-1][1], 2)

    def test_find_similar_single_item(self):
        """Test find_similar with a single item in haystack."""
        needle = [1.0, 0.0]
        haystack = [[0.5, 0.5]]
        results = find_similar(needle, haystack)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 0)

    def test_find_similar_all_identical(self):
        """Test find_similar when all vectors in haystack are identical."""
        needle = [1.0, 0.0]
        haystack = [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        results = find_similar(needle, haystack)
        # All should have same similarity score
        scores = [r[0] for r in results]
        self.assertTrue(all(abs(s - scores[0]) < 0.0001 for s in scores))


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

    def test_multiple_emojis(self):
        """Test removing multiple emojis."""
        text = "\U0001F600\U0001F601\U0001F602 Laughing!"
        cleaned = EMOJI_PATTERN.sub("", text)
        self.assertEqual(cleaned, " Laughing!")

    def test_emoji_at_end(self):
        """Test removing emoji at end of text."""
        text = "Great job!\U0001F44D"
        cleaned = EMOJI_PATTERN.sub("", text)
        self.assertEqual(cleaned, "Great job!")

    def test_emoji_only_text(self):
        """Test text with only emojis becomes empty."""
        text = "\U0001F600\U0001F601"
        cleaned = EMOJI_PATTERN.sub("", text)
        self.assertEqual(cleaned, "")

    def test_emojis_interspersed(self):
        """Test removing emojis interspersed in text."""
        text = "Hello\U0001F600World\U0001F601Test"
        cleaned = EMOJI_PATTERN.sub("", text)
        self.assertEqual(cleaned, "HelloWorldTest")


class TestRetrievalMode(unittest.TestCase):
    """Tests for the RetrievalMode enum."""

    def test_retrieval_mode_values(self):
        """Test that RetrievalMode enum has correct values."""
        self.assertEqual(RetrievalMode.RAG.value, "rag")
        self.assertEqual(RetrievalMode.CAG.value, "cag")

    def test_retrieval_mode_comparison(self):
        """Test that RetrievalMode enum members can be compared."""
        self.assertEqual(RetrievalMode.RAG, RetrievalMode.RAG)
        self.assertNotEqual(RetrievalMode.RAG, RetrievalMode.CAG)

    def test_retrieval_mode_from_value(self):
        """Test creating RetrievalMode from string value."""
        self.assertEqual(RetrievalMode("rag"), RetrievalMode.RAG)
        self.assertEqual(RetrievalMode("cag"), RetrievalMode.CAG)


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

    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            model_chatbot="custom-model",
            retrieval_mode=RetrievalMode.CAG,
            rag_top_k=20,
            similarity_threshold=0.9
        )
        self.assertEqual(config.model_chatbot, "custom-model")
        self.assertEqual(config.retrieval_mode, RetrievalMode.CAG)
        self.assertEqual(config.rag_top_k, 20)
        self.assertEqual(config.similarity_threshold, 0.9)

    def test_config_all_fields(self):
        """Test that Config has all expected fields."""
        config = Config()
        expected_fields = [
            'model_chatbot', 'model_embedding', 'model_voice_transcription',
            'system_language', 'voice_name', 'voice_enabled',
            'retrieval_mode', 'rag_top_k', 'cag_max_chars',
            'similarity_threshold', 'memory_file', 'embeddings_dir',
            'voice_dir', 'translation_cache_file'
        ]
        for field in expected_fields:
            self.assertTrue(hasattr(config, field), f"Missing field: {field}")

    def test_config_voice_settings(self):
        """Test voice-related config settings."""
        config = Config()
        self.assertEqual(config.voice_name, "if_sara")
        self.assertEqual(config.voice_enabled, True)
        self.assertEqual(config.voice_dir, "voice")


class TestEmbeddingsPersistence(unittest.TestCase):
    """Tests for save_embeddings and load_embeddings functions."""

    def setUp(self):
        """Create a temporary directory for embeddings."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        embeddings = {
            "hash": "abc123",
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }
        save_embeddings("test.txt", embeddings, self.temp_dir)
        loaded = load_embeddings("test", self.temp_dir)
        self.assertEqual(loaded, embeddings)

    def test_load_nonexistent_embeddings(self):
        """Test loading embeddings that don't exist returns None."""
        result = load_embeddings("nonexistent", self.temp_dir)
        self.assertIsNone(result)

    def test_embeddings_directory_creation(self):
        """Test that embeddings directory is created if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "new_embeddings")
        embeddings = {"hash": "test", "embeddings": []}
        save_embeddings("test.txt", embeddings, new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_embeddings_hash_preserved(self):
        """Test that hash is correctly preserved in embeddings."""
        test_hash = hashlib.md5(b"test content").hexdigest()
        embeddings = {"hash": test_hash, "embeddings": [[1.0, 2.0]]}
        save_embeddings("test.txt", embeddings, self.temp_dir)
        loaded = load_embeddings("test", self.temp_dir)
        self.assertEqual(loaded["hash"], test_hash)


class TestTranslationCache(unittest.TestCase):
    """Tests for translation cache functions."""

    def setUp(self):
        """Create a temporary file for cache."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.cache_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.cache_path):
            os.unlink(self.cache_path)

    def test_load_empty_cache(self):
        """Test loading cache when file doesn't exist."""
        os.unlink(self.cache_path)  # Remove the file
        cache = load_translation_cache(self.cache_path)
        self.assertEqual(cache, {})

    def test_save_and_load_cache(self):
        """Test saving and loading translation cache."""
        cache = {
            "italian::hello": "ciao",
            "italian::goodbye": "arrivederci"
        }
        save_translation_cache(cache, self.cache_path)
        loaded = load_translation_cache(self.cache_path)
        self.assertEqual(loaded, cache)

    def test_cache_with_unicode(self):
        """Test cache handles unicode characters correctly."""
        cache = {
            "italian::hello world": "ciao mondo",
            "italian::café": "caffè"
        }
        save_translation_cache(cache, self.cache_path)
        loaded = load_translation_cache(self.cache_path)
        self.assertEqual(loaded, cache)

    def test_load_corrupted_cache(self):
        """Test loading corrupted cache file returns empty dict."""
        with open(self.cache_path, 'w') as f:
            f.write("not valid json{{{")
        cache = load_translation_cache(self.cache_path)
        self.assertEqual(cache, {})

    def test_cache_key_format(self):
        """Test expected cache key format."""
        # The expected format is "language::text"
        language = "italian"
        text = "hello"
        expected_key = f"{language}::{text}"
        self.assertEqual(expected_key, "italian::hello")


class TestHashGeneration(unittest.TestCase):
    """Tests for hash generation used in embedding caching."""

    def test_md5_hash_consistency(self):
        """Test that MD5 hash is consistent for same content."""
        content = "test content"
        hash1 = hashlib.md5(content.encode("utf-8")).hexdigest()
        hash2 = hashlib.md5(content.encode("utf-8")).hexdigest()
        self.assertEqual(hash1, hash2)

    def test_md5_hash_different_content(self):
        """Test that different content produces different hash."""
        hash1 = hashlib.md5(b"content1").hexdigest()
        hash2 = hashlib.md5(b"content2").hexdigest()
        self.assertNotEqual(hash1, hash2)

    def test_chunks_to_hash(self):
        """Test hashing joined chunks (as used in get_embeddings)."""
        chunks = ["paragraph one", "paragraph two", "paragraph three"]
        current_hash = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
        self.assertEqual(len(current_hash), 32)  # MD5 produces 32 hex chars


class TestSimilarityThreshold(unittest.TestCase):
    """Tests for similarity threshold logic used in memory deduplication."""

    def test_below_threshold_is_unique(self):
        """Test that similarity below threshold is considered unique."""
        threshold = 0.8
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.5, 0.5, 0.0]  # Partially similar
        similarity = compute_cosine_similarity(vec_a, vec_b)
        is_unique = similarity <= threshold
        self.assertTrue(is_unique)

    def test_above_threshold_is_duplicate(self):
        """Test that similarity above threshold is considered duplicate."""
        threshold = 0.8
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.99, 0.1, 0.0]  # Very similar
        similarity = compute_cosine_similarity(vec_a, vec_b)
        is_duplicate = similarity > threshold
        self.assertTrue(is_duplicate)

    def test_exact_threshold(self):
        """Test behavior at exact threshold value."""
        threshold = 0.8
        # A similarity of exactly 0.8 should be considered unique (<=)
        similarity = 0.8
        is_unique = similarity <= threshold
        self.assertTrue(is_unique)


if __name__ == "__main__":
    unittest.main()
