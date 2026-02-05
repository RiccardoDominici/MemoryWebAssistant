# MemoryWebAssistant

A fully local AI chatbot with persistent memory, voice I/O, and web search capabilities. All processing runs on your machine—no cloud services required.

## Features

- **Local LLM Inference**: Uses [Ollama](https://ollama.com/) to run language models locally (Gemma, Llama, Mistral, etc.)
- **Persistent Memory**: Automatically extracts and stores important information from conversations
- **Dual Retrieval Modes**: Choose between RAG (Retrieval-Augmented Generation) and CAG (Cache-Augmented Generation)
- **Voice Input/Output**: Speech-to-text via Whisper, text-to-speech via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- **Web Search**: Optional internet search integration with user confirmation
- **Multi-language Support**: Automatic translation with caching
- **Privacy-First**: No data leaves your computer

## Retrieval Modes

MemoryWebAssistant supports two memory retrieval strategies:

| Mode | Description | Best For |
|------|-------------|----------|
| **RAG** | Retrieves only the most relevant memory chunks using semantic similarity search (default: top 15) | Large memory files, specific queries |
| **CAG** | Loads all available memory into context (up to configurable limit) | Small memory files, comprehensive context |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Audio input/output devices (for voice features)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RiccardoDominici/MemoryWebAssistant.git
cd MemoryWebAssistant
```

### 2. Install Python Dependencies

```bash
pip install numpy ollama pillow soundfile simpleaudio sounddevice faster-whisper misaki kokoro-onnx
```

### 3. Download Ollama Models

```bash
# Download chatbot model
ollama pull gemma3n:e4b

# Download embedding model
ollama pull snowflake-arctic-embed2
```

You can substitute these with any compatible models available in Ollama.

### 4. Set Up Voice Files

Download the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) voice synthesis files and place them in the `voice/` directory:

```
voice/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

### 5. (Optional) Set Up Web Search

To enable internet search capabilities, install [smolagents](https://github.com/huggingface/smolagents):

```bash
pip install smolagents
```

## Usage

### Running the Chatbot

```bash
python chat_bot.py
```

On startup, the assistant will:
1. Display the current configuration
2. Load existing memories from `memory.txt`
3. Calibrate the microphone silence threshold
4. Begin listening for voice input

Speak naturally—the assistant will transcribe your speech, process it with context from memory, and respond both in text and voice.

### Running Tests

The project includes a comprehensive test suite with 42 unit tests covering all core functionality.

```bash
# Run all tests
python -m unittest test_chat_bot -v

# Run a specific test class
python -m unittest test_chat_bot.TestParseFile -v

# Run a specific test
python -m unittest test_chat_bot.TestCosineSimilarity.test_opposite_vectors -v
```

#### Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestParseFile` | 6 | File parsing, empty files, whitespace handling |
| `TestCosineSimilarity` | 5 | Vector similarity calculations |
| `TestFindSimilar` | 4 | Similarity search and sorting |
| `TestEmojiPattern` | 5 | Emoji removal from text |
| `TestRetrievalMode` | 3 | RAG/CAG enum validation |
| `TestConfig` | 4 | Configuration defaults and customization |
| `TestEmbeddingsPersistence` | 4 | Embedding save/load operations |
| `TestTranslationCache` | 5 | Translation caching with unicode support |
| `TestHashGeneration` | 3 | MD5 hashing for cache invalidation |
| `TestSimilarityThreshold` | 3 | Memory deduplication logic |

## Configuration

All settings are centralized in the `Config` dataclass at the top of `chat_bot.py`:

```python
@dataclass
class Config:
    # Model Configuration
    model_chatbot: str = "gemma3n:e4b"
    model_embedding: str = "snowflake-arctic-embed2"
    model_voice_transcription: str = "large-v3"

    # Language & Voice
    system_language: str = "italian"
    voice_name: str = "if_sara"
    voice_enabled: bool = True

    # Retrieval Configuration
    retrieval_mode: RetrievalMode = RetrievalMode.RAG
    rag_top_k: int = 15                    # Number of chunks for RAG
    cag_max_chars: int = 50000             # Max characters for CAG
    similarity_threshold: float = 0.8      # Memory deduplication threshold

    # File Paths
    memory_file: str = "memory.txt"
    embeddings_dir: str = "embeddings"
    voice_dir: str = "voice"
    translation_cache_file: str = "translations.json"
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_chatbot` | Ollama model for chat responses | `gemma3n:e4b` |
| `model_embedding` | Ollama model for embeddings | `snowflake-arctic-embed2` |
| `model_voice_transcription` | Whisper model size | `large-v3` |
| `system_language` | Language for responses and TTS | `italian` |
| `voice_name` | Voice profile for TTS | `if_sara` |
| `voice_enabled` | Enable/disable voice output | `True` |
| `retrieval_mode` | `RAG` or `CAG` | `RAG` |
| `rag_top_k` | Number of memory chunks for RAG | `15` |
| `cag_max_chars` | Maximum context size for CAG | `50000` |
| `similarity_threshold` | Threshold for memory deduplication | `0.8` |

### Switching Retrieval Modes

To use CAG instead of RAG, modify the config:

```python
CONFIG = Config(retrieval_mode=RetrievalMode.CAG)
```

## Project Structure

```
MemoryWebAssistant/
├── chat_bot.py           # Main application
├── agent_web.py          # Web search agent
├── test_chat_bot.py      # Unit tests (42 tests)
├── translations.json     # Translation cache
├── memory.txt            # Persistent memory storage
├── embeddings/           # Cached embedding vectors
│   └── memory.json
├── voice/                # Voice synthesis files
│   ├── kokoro-v1.0.onnx
│   ├── voices-v1.0.bin
│   └── tmp.wav          # Temporary recording file
└── README.md
```

## How It Works

1. **Voice Input**: Records audio until silence is detected, then transcribes using Whisper
2. **Memory Retrieval**: Fetches relevant context using RAG (semantic search) or CAG (full context)
3. **Web Search**: Optionally searches the internet if the query requires current information
4. **Response Generation**: Generates a response using the LLM with retrieved context
5. **Memory Extraction**: Automatically extracts and saves new information from the conversation
6. **Voice Output**: Synthesizes speech for each sentence as it's generated

## Privacy

All data is processed locally:
- Chat history stays on your machine
- Memory is stored in local text files
- Voice processing happens entirely offline
- Web search is optional and requires explicit user confirmation

## Troubleshooting

### Common Issues

**Ollama connection error**
```bash
# Ensure Ollama is running
ollama serve
```

**Missing voice files**
```
Ensure kokoro-v1.0.onnx and voices-v1.0.bin are in the voice/ directory
```

**Microphone not detected**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Tests failing with import errors**
```bash
# Tests are self-contained and don't require external dependencies
python -m unittest test_chat_bot -v
```

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request:

```bash
python -m unittest test_chat_bot -v
```

## Credits

- [Ollama](https://ollama.com) - Local LLM inference
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) - Text-to-speech
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Speech-to-text
- [smolagents](https://github.com/huggingface/smolagents) - Web search agent

## License

See individual component licenses in their respective repositories.
