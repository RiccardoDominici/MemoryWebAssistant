# MemoryWebAssistant: Local AI Chatbot with Voice

## Overview
This project is a fully local AI chatbot assistant with memory, voice output, and optional internet search capabilities. All processing, including language model inference and text-to-speech (TTS), runs on your own machine—no cloud services required.

- **Chatbot Model:** Uses [Ollama](https://ollama.com/) to run LLMs locally (e.g., Gemma, Llama, etc.).
- **Embeddings:** Uses Ollama's embedding models for memory/context retrieval.
- **Voice Synthesis:** Uses [kokoro_onnx](https://github.com/thewh1teagle/kokoro-onnx) for local TTS in your chosen language.
- **No data leaves your computer.**

---

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install numpy ollama pillow soundfile simpleaudio
```

### 2. Download Ollama and Required Models
- Install [Ollama](https://ollama.com/download) for your OS.
- Download the chatbot model you want (e.g., Gemma, Llama, etc.):
  ```bash
  ollama pull gemma:4b
  ollama pull nomic-embed-text
  ```
- You can use any supported model; update `MODEL_CHATBOT` and `MODEL_EMB` in `chat_bot.py` as needed.

### 3. Download kokoro_onnx Voice Templates
- Download the [kokoro_onnx](https://github.com/r9y9/kokoro-onnx) voice synthesis engine and voices.
- Place the following files in the `voice/` directory:
  - `kokoro-v1.0.onnx`
  - `voices-v1.0.bin`
- For other languages/voices, download the appropriate entries from the kokoro_onnx repository and place them in `voice/`.

### 4. Add Web Search Agent (smolagent)
- To enable optional internet search, clone and set up [smolagent](https://github.com/huggingface/smolagents) as described in their repository. Follow their instructions to install and configure the agent for your system.

---

## Usage

Run the chatbot locally:

```bash
python chat_bot.py
```

- The assistant will respond to your input, use memory, and speak responses aloud.
- All computation is local. No internet connection is required except for optional web search (with your permission).

---

## Customization

You can easily customize the assistant's behavior and language by editing the following variables at the top of `chat_bot.py`:

```python
MODEL_CHATBOT = "gemma3:4b"      # Name of the chatbot model to use
MODEL_EMB = "nomic-embed-text"   # Name of the model for embeddings
SYSTEM_LANGUAGE = "italian"      # System language for translation
VOICE = "if_sara"                # Default voice for the assistant
VOICE_TOGGLED = True              # Enable/disable voice output
```

- **Change the chatbot model:** Set `MODEL_CHATBOT` to any model you have downloaded with Ollama (e.g., `llama2:7b`, `mistral:7b`, etc.).
- **Change the embedding model:** Set `MODEL_EMB` to any supported embedding model (e.g., `nomic-embed-text`).
- **Change the system language:** Set `SYSTEM_LANGUAGE` to your preferred language (e.g., `english`, `italian`, etc.).
- **Change the voice:** Set `VOICE` to any available voice in your kokoro_onnx voices directory.
- **Enable/disable voice output:** Set `VOICE_TOGGLED` to `True` or `False`.

---

## Privacy
**All data, including chat history, memory, and voice synthesis, is processed locally. No data is sent to any external server by default.**

---

## Credits
- [Ollama](https://ollama.com/)
- [kokoro_onnx](https://github.com/r9y9/kokoro-onnx)
- [smolagent](https://github.com/smol-ai/smolagent)

---

## License
See individual component licenses in their respective repositories.
