# ==============================================================================
# MemoryWebAssistant - AI Chatbot with Memory, Voice & Web Search
# ==============================================================================
"""
A local AI chatbot with persistent memory, voice I/O, and web search capabilities.
Supports both RAG (Retrieval-Augmented Generation) and CAG (Cache-Augmented Generation).
"""

# ==============================================================================
# Standard Library Imports
# ==============================================================================
import os
import sys
import json
import re
import queue
import threading
import asyncio
import datetime
import base64
import hashlib
import time
import shutil
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# ==============================================================================
# Third-Party Imports
# ==============================================================================
import numpy as np
from numpy.linalg import norm
import ollama
from PIL import ImageGrab
import soundfile as sf
import simpleaudio as sa
from misaki.espeak import EspeakG2P
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel
import sounddevice as sd

# ==============================================================================
# Local Imports
# ==============================================================================
import agent_web


# ==============================================================================
# Configuration - Retrieval Mode
# ==============================================================================
class RetrievalMode(Enum):
    """
    Defines the retrieval strategy for memory context.

    RAG (Retrieval-Augmented Generation):
        - Retrieves only the most relevant chunks based on semantic similarity
        - More efficient for large memory files
        - Uses top-K similarity search (default K=15)
        - Better when memory is large and queries are specific

    CAG (Cache-Augmented Generation):
        - Loads ALL available memory into the context
        - Better comprehension of full context
        - No information loss from retrieval
        - Best when memory is small enough to fit in context window
        - Maximum context size is configurable via CAG_MAX_TOKENS
    """
    RAG = "rag"
    CAG = "cag"


# ==============================================================================
# Configuration - Main Settings
# ==============================================================================
@dataclass
class Config:
    """Central configuration for the chatbot."""

    # Model Configuration
    model_chatbot: str = "gemma3n:e4b"
    model_embedding: str = "snowflake-arctic-embed2"
    model_voice_transcription: str = "large-v3"

    # Language & Voice
    system_language: str = "italian"
    voice_name: str = "if_sara"
    voice_enabled: bool = True

    # Retrieval Configuration
    retrieval_mode: RetrievalMode = RetrievalMode.RAG  # Change to CAG for full context
    rag_top_k: int = 15                                # Number of chunks for RAG
    cag_max_chars: int = 50000                         # Max characters for CAG context
    similarity_threshold: float = 0.8                  # Threshold for memory deduplication

    # File Paths
    memory_file: str = "memory.txt"
    embeddings_dir: str = "embeddings"
    voice_dir: str = "voice"
    translation_cache_file: str = "translations.json"


# Global configuration instance - modify this to change settings
CONFIG = Config()


# ==============================================================================
# Global State
# ==============================================================================
audio_queue: queue.Queue = queue.Queue()
generated_audio_queue: queue.Queue = queue.Queue()
conversation: list = []
translation_cache: dict = {}


# ==============================================================================
# Emoji Pattern (Compiled Once)
# ==============================================================================
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)


# ==============================================================================
# Translation Cache Functions
# ==============================================================================
def load_translation_cache() -> None:
    """Load the translation cache from disk."""
    global translation_cache
    if os.path.exists(CONFIG.translation_cache_file):
        try:
            with open(CONFIG.translation_cache_file, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            translation_cache = {}
    else:
        translation_cache = {}


def save_translation_cache() -> None:
    """Save the translation cache to disk."""
    with open(CONFIG.translation_cache_file, "w", encoding="utf-8") as f:
        json.dump(translation_cache, f, ensure_ascii=False, indent=2)


def auto_translate(text: str) -> str:
    """
    Translate text to the configured system language.
    Uses cache to avoid redundant API calls.

    Args:
        text: The text to translate (assumed English)

    Returns:
        Translated text in the system language
    """
    if CONFIG.system_language.lower() == "english":
        return text

    cache_key = f"{CONFIG.system_language.lower()}::{text}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    translate_prompt = (
        f"Translate the following text into {CONFIG.system_language}: '{text}'\n"
        f"Generate only the translated text in {CONFIG.system_language}."
    )

    response = ollama.chat(
        model=CONFIG.model_chatbot,
        messages=[{"role": "system", "content": translate_prompt}],
        options={"temperature": 0},
        stream=False
    )

    translated = response["message"]["content"]
    translation_cache[cache_key] = translated
    save_translation_cache()

    return translated


# ==============================================================================
# File & Embedding Functions
# ==============================================================================
def parse_file(filename: str) -> list[str]:
    """
    Parse a file into paragraphs separated by blank lines.

    Args:
        filename: Path to the file to parse

    Returns:
        List of paragraph strings
    """
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


def save_embeddings(filename: str, embeddings: dict) -> None:
    """
    Save embeddings to a JSON file in the embeddings directory.

    Args:
        filename: Original file name (used to derive embedding file name)
        embeddings: Dictionary containing hash and embedding vectors
    """
    name = filename.split(".")[0]
    os.makedirs(CONFIG.embeddings_dir, exist_ok=True)

    with open(f"{CONFIG.embeddings_dir}/{name}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename: str) -> Optional[dict]:
    """
    Load embeddings from a JSON file if it exists.

    Args:
        filename: Original file name

    Returns:
        Embeddings dictionary or None if not found
    """
    path = f"{CONFIG.embeddings_dir}/{filename}.json"
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_embeddings(filename: str, chunks: list[str]) -> list[list[float]]:
    """
    Generate or load cached embeddings for text chunks.
    Uses MD5 hash to detect if the file has changed.

    Args:
        filename: Source file name for caching
        chunks: List of text chunks to embed

    Returns:
        List of embedding vectors
    """
    current_hash = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
    loaded = load_embeddings(filename)

    if loaded is not None and loaded.get("hash") == current_hash:
        return loaded["embeddings"]

    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = [
        ollama.embeddings(model=CONFIG.model_embedding, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    data_to_save = {"hash": current_hash, "embeddings": embeddings}
    save_embeddings(filename, data_to_save)

    return embeddings


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def find_similar(needle: list[float], haystack: list[list[float]]) -> list[tuple[float, int]]:
    """
    Find vectors in haystack similar to needle using cosine similarity.

    Args:
        needle: Query embedding vector
        haystack: List of embedding vectors to search

    Returns:
        Sorted list of (similarity_score, index) tuples, highest first
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item))
        for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


# ==============================================================================
# Memory Retrieval Functions (RAG & CAG)
# ==============================================================================
def retrieve_memory_context_rag(
    embeddings: list[list[float]],
    paragraphs: list[str],
    prompt: str
) -> int:
    """
    RAG: Retrieve relevant memory context using semantic similarity search.
    Adds only the top-K most relevant paragraphs to the conversation.

    Args:
        embeddings: List of paragraph embeddings
        paragraphs: List of memory paragraphs
        prompt: User's input prompt

    Returns:
        Index of the added context message in conversation
    """
    prompt_embedding = ollama.embeddings(
        model=CONFIG.model_embedding,
        prompt=prompt
    )["embedding"]

    most_similar = find_similar(prompt_embedding, embeddings)[:CONFIG.rag_top_k]

    # Build context with timestamp
    context_parts = [
        "\nContext (RAG - Top {} relevant memories):".format(CONFIG.rag_top_k),
        auto_translate("Current Date and Time: ") +
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ""
    ]

    for score, index in most_similar:
        context_parts.append(f"[Relevance: {score:.2f}] {paragraphs[index]}")

    context = "\n".join(context_parts)
    conversation.append({"role": "system", "content": context})

    return len(conversation) - 1


def retrieve_memory_context_cag(paragraphs: list[str]) -> int:
    """
    CAG: Load all available memory context into the conversation.
    Loads the full memory up to a configurable character limit.

    Args:
        paragraphs: List of all memory paragraphs

    Returns:
        Index of the added context message in conversation
    """
    # Build context with timestamp
    context_parts = [
        "\nContext (CAG - Full Memory Cache):",
        auto_translate("Current Date and Time: ") +
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"Total memories: {len(paragraphs)}",
        ""
    ]

    # Add all paragraphs up to the character limit
    current_length = sum(len(p) for p in context_parts)
    included_count = 0

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 2 > CONFIG.cag_max_chars:
            context_parts.append(
                f"\n[... {len(paragraphs) - included_count} more memories truncated due to size limit ...]"
            )
            break
        context_parts.append(paragraph)
        current_length += len(paragraph) + 1
        included_count += 1

    context = "\n".join(context_parts)
    conversation.append({"role": "system", "content": context})

    return len(conversation) - 1


def retrieve_memory_context(
    embeddings: list[list[float]],
    paragraphs: list[str],
    prompt: str
) -> int:
    """
    Retrieve memory context using the configured retrieval mode.

    Args:
        embeddings: List of paragraph embeddings
        paragraphs: List of memory paragraphs
        prompt: User's input prompt

    Returns:
        Index of the added context message in conversation
    """
    if CONFIG.retrieval_mode == RetrievalMode.CAG:
        return retrieve_memory_context_cag(paragraphs)
    else:
        return retrieve_memory_context_rag(embeddings, paragraphs, prompt)


# ==============================================================================
# Memory Saving Functions
# ==============================================================================
def save_information_from_conversation(
    prompt: str,
    response_ai: str,
    filename: str,
    embeddings: list[list[float]],
    paragraphs: list[str]
) -> tuple[list[list[float]], list[str]]:
    """
    Extract and save new memories from the conversation.
    Deduplicates using similarity threshold to avoid storing redundant info.

    Args:
        prompt: User's input
        response_ai: AI's response
        filename: Memory file path
        embeddings: Current embedding vectors
        paragraphs: Current memory paragraphs

    Returns:
        Updated (embeddings, paragraphs) tuple
    """
    system_prompt = {
        "role": "system",
        "content": auto_translate(
            "You are a memory generator. Given the conversation, extract important "
            "information that the assistant should remember for future interactions. "
            "Respond with one or more concise sentences, each on a new line. "
            "Only extract factual, useful information - not conversational pleasantries."
        )
    }

    temp_conversation = [
        system_prompt,
        {"role": "user", "content": f"User: {prompt}\nAssistant: {response_ai}"}
    ]

    response = ollama.chat(
        model=CONFIG.model_chatbot,
        messages=temp_conversation,
        options={"temperature": 0},
        stream=False
    )

    memories = [m.strip() for m in response["message"]["content"].split('\n') if m.strip()]

    if not memories:
        return embeddings, paragraphs

    # Generate embeddings for candidate memories
    candidate_embeddings = [
        ollama.embeddings(model=CONFIG.model_embedding, prompt=m)["embedding"]
        for m in memories
    ]

    new_embeddings = []
    new_memories = []

    for mem_text, candidate_emb in zip(memories, candidate_embeddings):
        # Check similarity with existing memories
        if embeddings:
            similarities = [
                compute_cosine_similarity(candidate_emb, existing)
                for existing in embeddings
            ]
            max_similarity = max(similarities)
        else:
            max_similarity = 0

        # Only save if sufficiently different from existing memories
        if max_similarity <= CONFIG.similarity_threshold:
            new_embeddings.append(candidate_emb)
            new_memories.append(mem_text)

    # Append new memories to file
    if new_memories:
        with open(filename, "a", encoding="utf-8") as f:
            for mem in new_memories:
                f.write("\n\n" + mem)

        paragraphs.extend(new_memories)
        embeddings.extend(new_embeddings)
        print(f"[Memory] Saved {len(new_memories)} new memories")

    return embeddings, paragraphs


# ==============================================================================
# Response Generation
# ==============================================================================
def stream_response(prompt: str) -> str:
    """
    Generate and stream a response from the chatbot.
    Plays audio for each complete sentence.

    Args:
        prompt: User's input

    Returns:
        Complete response text
    """
    conversation.append({"role": "user", "content": prompt})

    response = ""
    stream = ollama.chat(
        model=CONFIG.model_chatbot,
        messages=conversation,
        stream=True
    )

    print("<Gemma> ", end="", flush=True)

    sentence_buffer = ""
    for chunk in stream:
        content = EMOJI_PATTERN.sub("", chunk["message"]["content"])
        sentence_buffer += content
        response += content
        print(content, end="", flush=True)

        # Play audio when we have a complete sentence
        if content.endswith((".", "?", "!", ":")) and len(sentence_buffer) > 30:
            play_audio(sentence_buffer)
            sentence_buffer = ""

    # Play any remaining text
    if len(sentence_buffer) > 1:
        play_audio(sentence_buffer)

    conversation.append({"role": "assistant", "content": response})
    print("")

    return response


# ==============================================================================
# Screenshot Handling
# ==============================================================================
def handle_screenshot_request(prompt: str) -> int:
    """
    Determine if a screenshot is needed and capture it if so.

    Args:
        prompt: User's input

    Returns:
        Index of screenshot in conversation, or -1 if not needed
    """
    temp_conversation = [
        {
            "role": "system",
            "content": auto_translate(
                "You are not an AI assistant. Your only task is to decide if the "
                "user prompt requires a screenshot for context. Respond only with "
                "\"True\" or \"False\"."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = ollama.chat(
        model=CONFIG.model_chatbot,
        messages=temp_conversation,
        options={"temperature": 0}
    )

    if "true" in response["message"]["content"].lower():
        screenshot = ImageGrab.grab()
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        conversation.append({"role": "assistant", "images": [image_base64]})
        print("[Screenshot captured]")
        return len(conversation) - 1

    return -1


# ==============================================================================
# Web Search Handling
# ==============================================================================
def handle_search_request(prompt: str) -> int:
    """
    Determine if a web search is needed and perform it with user confirmation.

    Args:
        prompt: User's input

    Returns:
        Index of search results in conversation, or -1 if not performed
    """
    temp_conversation = [
        {
            "role": "system",
            "content": auto_translate(
                "You are not an AI assistant. Decide if the user prompt requires "
                "an internet search to answer properly. Reply only \"True\" for "
                "a search or \"False\" if not needed."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = ollama.chat(
        model=CONFIG.model_chatbot,
        messages=temp_conversation,
        options={"temperature": 0}
    )

    if "true" not in response["message"]["content"].lower():
        return -1

    # Ask user for confirmation
    confirm_message = auto_translate(
        "I can look it up on the internet to answer but it will take a long time, "
        "shall I continue?"
    )
    play_audio(confirm_message)

    user_response = input(f"<Gemma> {confirm_message} (y/n)\n<You> ").strip().lower()

    if "y" not in user_response:
        return -1

    # Perform search
    search_message = auto_translate("Performing internet search...")
    print(f"<Gemma> {search_message}")
    play_audio(search_message)

    search_results = agent_web.run_agent(prompt, step=1)

    conversation.append({
        "role": "assistant",
        "content": (
            auto_translate("I performed an online search. Here are the results:") +
            f"\n{search_results}\n" +
            auto_translate("Now I can answer the user's question:") +
            f"\n{prompt}"
        )
    })

    return len(conversation) - 1


# ==============================================================================
# Audio Functions
# ==============================================================================
def play_audio(text: str) -> None:
    """
    Queue text for audio playback if voice is enabled.

    Args:
        text: Text to speak
    """
    if not CONFIG.voice_enabled:
        return

    # Clean text for TTS
    clean_text = text.replace("*", "")
    clean_text = EMOJI_PATTERN.sub("", clean_text)

    if clean_text.strip():
        audio_queue.put(clean_text)


def audio_generator() -> None:
    """
    Background thread that generates audio samples from queued text.
    Uses Kokoro TTS with eSpeak phonemizer.
    """
    kokoro = Kokoro(
        f"{CONFIG.voice_dir}/kokoro-v1.0.onnx",
        f"{CONFIG.voice_dir}/voices-v1.0.bin"
    )

    while True:
        text = audio_queue.get()

        # Generate phonemes
        g2p = EspeakG2P(language=CONFIG.system_language[:2])
        phonemes, _ = g2p(text)

        # Generate audio
        samples, sample_rate = kokoro.create(phonemes, CONFIG.voice_name, is_phonemes=True)

        num_channels = samples.shape[1] if samples.ndim > 1 else 1
        bytes_per_sample = samples.dtype.itemsize

        generated_audio_queue.put((samples, sample_rate, num_channels, bytes_per_sample))
        audio_queue.task_done()


def audio_worker() -> None:
    """
    Background thread that plays generated audio samples.
    """
    while True:
        samples, sample_rate, num_channels, bytes_per_sample = generated_audio_queue.get()

        wave_obj = sa.WaveObject(
            samples.tobytes(),
            num_channels,
            bytes_per_sample,
            sample_rate
        )
        wave_obj.play().wait_done()

        generated_audio_queue.task_done()


# ==============================================================================
# Voice Input Functions
# ==============================================================================
def record_until_silence(
    silence_threshold: float = 0.2,
    silence_duration: float = 2.0,
    sample_rate: int = 44100
) -> bool:
    """
    Record audio until silence is detected for the specified duration.

    Args:
        silence_threshold: RMS threshold below which is considered silence
        silence_duration: Seconds of silence to trigger stop
        sample_rate: Audio sample rate

    Returns:
        True if audio was recorded, False otherwise
    """
    # Wait for TTS to finish
    audio_queue.join()
    generated_audio_queue.join()

    buffer = []
    last_sound_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal last_sound_time
        volume_norm = np.sqrt(np.mean(indata ** 2))
        buffer.append(indata.copy())
        if volume_norm > silence_threshold:
            last_sound_time = time.time()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        max_dots = shutil.get_terminal_size().columns - len("<You> ") - 1
        max_dots = max(max_dots, 1)

        while True:
            elapsed_silence = time.time() - last_sound_time

            if elapsed_silence > silence_duration:
                # Clear the visual indicator
                sys.stdout.write("\r" + " " * (len("<You> ") + max_dots) + "\r")
                sys.stdout.flush()
                break

            # Show visual feedback
            ratio = elapsed_silence / silence_duration
            dot_count = max(int((1 - ratio) * max_dots), 1)
            dots = "." * dot_count
            sys.stdout.write(f"\r<You> {dots}" + " " * (max_dots - dot_count))
            sys.stdout.flush()

            time.sleep(0.1)

    if not buffer:
        return False

    # Save recorded audio
    audio_data = np.concatenate(buffer, axis=0)
    sf.write(f"{CONFIG.voice_dir}/tmp.wav", audio_data, sample_rate)

    return True


def get_input_from_microphone(model: WhisperModel, silence_threshold: float) -> str:
    """
    Record voice input and transcribe it.

    Args:
        model: WhisperModel instance for transcription
        silence_threshold: Calibrated silence threshold

    Returns:
        Transcribed text
    """
    while not record_until_silence(
        silence_threshold=silence_threshold,
        silence_duration=2,
        sample_rate=44100
    ):
        time.sleep(0.1)

    print("\r<You> ", end="", flush=True)

    segments, _ = model.transcribe(
        f"{CONFIG.voice_dir}/tmp.wav",
        word_timestamps=True,
        vad_filter=True
    )

    output_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                output_words.append(word.word)
                print(word.word, end="", flush=True)

    output = " ".join(output_words).strip()
    if output:
        print("")

    return output


def calibrate_silence_threshold(
    duration: float = 5.0,
    sample_rate: int = 44100
) -> float:
    """
    Calibrate the silence threshold by measuring ambient noise.

    Args:
        duration: Duration in seconds to record for calibration
        sample_rate: Audio sample rate

    Returns:
        Calibrated silence threshold value
    """
    print(auto_translate("Calibrating silence threshold. Please remain quiet..."))

    recordings = []

    def callback(indata, frames, time_info, status):
        recordings.append(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        sd.sleep(int(duration * 1000))

    if not recordings:
        print("No data recorded during calibration.")
        return 0.2  # Fallback threshold

    audio_data = np.concatenate(recordings, axis=0)

    # Compute RMS for each frame
    frame_size = 1024
    rms_values = []

    for start in range(0, len(audio_data), frame_size):
        frame = audio_data[start:start + frame_size]
        if frame.size > 0:
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)

    mean_rms = np.mean(rms_values)
    threshold = mean_rms * 2 + 0.01  # 2x mean + small offset

    print(f"Calibrated silence threshold: {threshold:.4f}")
    os.system("cls" if os.name == "nt" else "clear")

    return threshold


# ==============================================================================
# Main Application
# ==============================================================================
def print_config_info() -> None:
    """Print current configuration information."""
    mode_info = {
        RetrievalMode.RAG: f"RAG (Top {CONFIG.rag_top_k} chunks)",
        RetrievalMode.CAG: f"CAG (Full context, max {CONFIG.cag_max_chars} chars)"
    }

    print("=" * 60)
    print("MemoryWebAssistant Configuration")
    print("=" * 60)
    print(f"  Retrieval Mode: {mode_info[CONFIG.retrieval_mode]}")
    print(f"  Model: {CONFIG.model_chatbot}")
    print(f"  Language: {CONFIG.system_language}")
    print(f"  Voice: {CONFIG.voice_enabled}")
    print("=" * 60)
    print()


async def main() -> None:
    """
    Main application loop.
    Handles voice input, memory retrieval, web search, and response generation.
    """
    load_translation_cache()
    print_config_info()

    # Initialize system prompt
    system_prompt = auto_translate(
        "You are Gemma, an AI assistant with access to memories and internet search. "
        "Respond concisely and expressively with proper punctuation (no emojis)."
    )
    conversation.append({"role": "system", "content": system_prompt})

    # Initialize memory file
    if not os.path.exists(CONFIG.memory_file):
        with open(CONFIG.memory_file, "w", encoding="utf-8") as f:
            f.write("")

    paragraphs = parse_file(CONFIG.memory_file)
    embeddings = get_embeddings(CONFIG.memory_file, paragraphs)

    print(f"[Memory] Loaded {len(paragraphs)} memory entries")

    # Start audio threads
    threading.Thread(target=audio_generator, daemon=True).start()
    threading.Thread(target=audio_worker, daemon=True).start()

    # Initialize voice transcription
    model = WhisperModel(
        CONFIG.model_voice_transcription,
        device="cpu",
        compute_type="int8"
    )
    silence_threshold = calibrate_silence_threshold()

    # Main conversation loop
    while True:
        prompt = get_input_from_microphone(model, silence_threshold)

        if not prompt:
            continue

        # Retrieve context based on configured mode
        memory_idx = retrieve_memory_context(embeddings, paragraphs, prompt)

        # Check for web search needs
        search_idx = handle_search_request(prompt)

        # Generate response
        bot_response = stream_response(prompt)

        # Clean up temporary context messages
        if search_idx >= 0:
            conversation.pop(search_idx)
        if memory_idx >= 0:
            # Adjust index if we already removed the search result
            adjusted_idx = memory_idx if search_idx < 0 else memory_idx
            if adjusted_idx < len(conversation):
                conversation.pop(adjusted_idx)

        # Save new memories from conversation
        embeddings, paragraphs = save_information_from_conversation(
            prompt, bot_response, CONFIG.memory_file, embeddings, paragraphs
        )


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    asyncio.run(main())
