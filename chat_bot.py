# =====================
# Standard Library Imports
# =====================
import os
import json
import re
import queue
import threading
import asyncio
import datetime
import base64
import hashlib
from io import BytesIO

# =====================
# Third-Party Imports
# =====================
import numpy as np
from numpy.linalg import norm
import ollama
from PIL import Image, ImageGrab
import soundfile as sf  
import simpleaudio as sa
from misaki import espeak 
from misaki.espeak import EspeakG2P 
from kokoro_onnx import Kokoro 
import agent_web

# =====================
# Global Variables & Constants
# =====================
MODEL_CHATBOT = "gemma3:4b"  # Chatbot model name
MODEL_EMB = "nomic-embed-text"  # Embedding model name

SYSTEM_LANGUAGE = "italian"  # System language for translation
VOICE = "if_sara"  # Default TTS voice
VOICE_TOGGLED = True   # Whether voice output is enabled

audio_queue = queue.Queue()  # Queue for audio playback requests
conversation = []  # History of conversation messages

# =====================
# Utility Functions
# =====================
def auto_translate(text):
    """
    Translates text into the system language (if not English).
    """
    if SYSTEM_LANGUAGE.lower() == 'english':
        return text
    translate_prompt = (
        f"Translate the following text into {SYSTEM_LANGUAGE}: '{text}'\n"
        f"Generate only the translated text in {SYSTEM_LANGUAGE}."
    )
    response = ollama.chat(
        model=MODEL_CHATBOT,
        messages=[{"role": "system", "content": translate_prompt}],
        options={"temperature": 0},
        stream=False
    )
    return response["message"]["content"]

def parse_file(filename):
    """
    Reads a file and splits its content into paragraphs.
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

def save_embeddings(filename, embeddings):
    """
    Saves embeddings to a JSON file in the 'embeddings' directory.
    """
    name = filename.split(".")[0]
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{name}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    """
    Loads embeddings from a JSON file if it exists.
    """
    path = f"embeddings/{filename}.json"
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        return json.load(f)

def get_embeddings(filename, chunks):
    """
    Generates or loads embeddings for each text chunk using a file hash.
    """
    current_hash = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
    loaded = load_embeddings(filename)
    if loaded is not False and loaded.get("hash") == current_hash:
        return loaded["embeddings"]
    embeddings = [ollama.embeddings(model=MODEL_EMB, prompt=chunk)["embedding"] for chunk in chunks]
    data_to_save = {"hash": current_hash, "embeddings": embeddings}
    save_embeddings(filename, data_to_save)
    return embeddings

def find_similar(needle, haystack):
    """
    Finds vectors in haystack similar to needle using cosine similarity.
    Returns a sorted list of (score, index) tuples.
    """
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# =====================
# Core Chatbot Functions
# =====================
def stream_response(prompt):
    """
    Streams response from the chatbot: prints to console and plays audio.
    """
    conversation.append({"role": "user", "content": prompt})
    response = ''
    stream = ollama.chat(model=MODEL_CHATBOT, messages=conversation, stream=True)
    print("<Gemma> ", end='', flush=True)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    tmp_content = ''
    for chunk in stream:
        content = emoji_pattern.sub("", chunk["message"]["content"])
        tmp_content += content
        if (content.endswith(('.', '?', '!', ':')) and len(tmp_content) > 60):
            play_audio(tmp_content)
            tmp_content = ''
        response += content
        print(content, end='', flush=True)
    if tmp_content:
        play_audio(tmp_content)
    conversation.append({"role": "assistant", "content": response})
    print('')
    return response

def handle_screenshot_request(prompt):
    """
    Determines whether a screenshot is required based on the prompt.
    Adds the screenshot (if taken) to the conversation.
    """
    temp_conversation = [
        {"role": "system", "content": auto_translate(
            "You are not an AI assistant. Your only task is to decide if the last user prompt requires a screenshot for context. Respond with \"True\" or \"False\".")
        },
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=MODEL_CHATBOT, messages=temp_conversation, options={"temperature": 0})
    response = response["message"]["content"]
    if "true" in response.lower():
        screenshot = ImageGrab.grab()
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        conversation.append({"role": "assistant", "images": [str(image_base64)]})
        print("Taking screenshot...")
        return len(conversation) - 1
    return -1

def retrieve_memory_context(embeddings, paragraphs, prompt):
    """
    Retrieves relevant memory context for the prompt and appends it to the conversation.
    """
    prompt_embedding = ollama.embeddings(model=MODEL_EMB, prompt=prompt)["embedding"]
    most_similar_chunks = find_similar(prompt_embedding, embeddings)[:15]
    mem_prompt = "\nContext: " + auto_translate("Current Date and Time: ") + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    for score, index in most_similar_chunks:
        if score > 0.5:
            mem_prompt += paragraphs[index] + "\n"
    conversation.append({"role": "system", "content": mem_prompt})
    return len(conversation) - 1

def save_information_from_index(prompt, response_ai, filename, embeddings, paragraphs):
    """
    Extracts memories from the conversation and saves them,
    also updating the memory embeddings and paragraphs.
    """
    system_prompt = {
        "role": "system",
        "content": auto_translate(
            ("You are a memory generator. Given the full conversation, extract data that the assistant must remember. "
             "Respond with one or more concise sentences (each on a new line) without additional explanation.")
        )
    }
    temp_conversation = [system_prompt, {"role": "user", "content": f"user: {prompt}\nAI: {response_ai}"}]
    response = ollama.chat(model=MODEL_CHATBOT, messages=temp_conversation, options={"temperature": 0}, stream=False)
    response = response["message"]["content"]
    memories = response.split('\n')
    if memories == [""]:
        return embeddings, paragraphs
    candidate_embeddings = [ollama.embeddings(model=MODEL_EMB, prompt=m)["embedding"] for m in memories if m.strip()]
    new_embeddings = []
    threshold = 0.8
    with open(filename, "a", encoding="utf-8") as f:
        for mem_text, candidate in zip(memories, candidate_embeddings):
            similarities = [np.dot(candidate, existing) / (norm(candidate) * norm(existing)) for existing in embeddings]
            if max(similarities, default=0) > threshold:
                continue
            new_embeddings.append(candidate)
            f.write("\n\n" + mem_text)
            paragraphs.append(mem_text)
    embeddings.extend(new_embeddings)
    return embeddings, paragraphs

def handle_search_request(prompt):
    """
    Checks if an internet search is needed for the prompt.
    If needed, asks the user for confirmation and performs the search.
    """
    temp_conversation = [
        {"role": "system", "content": auto_translate(
            ("You are not an AI assistant. Decide if the last prompt requires an internet search. "
             "Reply \"True\" for a search or \"False\" if not needed.")
        )},
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=MODEL_CHATBOT, messages=temp_conversation, options={"temperature": 0})
    response = response["message"]["content"]
    if "true" in response.lower():
        gemma_fixed_response = auto_translate("I can look it up on the internet to answer but it will take a long time, shall I continue?")
        play_audio(gemma_fixed_response)
        request_search = input(f"<Gemma> {gemma_fixed_response} (y/n)\n<You> ").strip().lower()
        if "y" in request_search:
            gemma_fixed_response = auto_translate("I perform the Internet search...")
            print(f"<Gemma> {gemma_fixed_response}")
            play_audio(gemma_fixed_response)
            response_search = agent_web.run_agent(prompt, step=1)
            conversation.append({
                "role": "assistant",
                "content": auto_translate("I did an online search to find the user's answer. Here are the results:") +
                           "\n" + auto_translate(str(response_search)) +
                           "\n" + auto_translate("Now I can answer the user's question:") + "\n" + prompt
            })
        return len(conversation) - 1
    return -1

# =====================
# Audio Playback Functions
# =====================
def play_audio(text):
    """
    Enqueues text for audio playback if voice output is enabled.
    """
    if VOICE_TOGGLED:
        audio_queue.put(text)

def audio_worker():
    """
    Processes queued audio playback requests using TTS.
    """
    kokoro = Kokoro("voice/kokoro-v1.0.onnx", "voice/voices-v1.0.bin")
    while True:
        text = audio_queue.get()
        g2p = EspeakG2P(language="it")
        phonemes, _ = g2p(text)
        samples, sample_rate = kokoro.create(phonemes, VOICE, is_phonemes=True)
        sf.write("voice/tmp.wav", samples, sample_rate)
        wave_obj = sa.WaveObject.from_wave_file("voice/tmp.wav")
        wave_obj.play().wait_done()
        audio_queue.task_done()

# =====================
# Main Application Entry Point
# =====================
async def main():
    """
    Main loop: Handles user input while retrieving context, search results, and processing responses.
    """
    SYSTEM_PROMPT = auto_translate(
        ("You are Gemma, an AI assistant that retrieves live data via internet, memory, or screenshots. "
         "Incorporate any attached results before responding concisely and expressively with abundant punctuation (no emojis).")
    )
    conversation.append({"role": "system", "content": SYSTEM_PROMPT})
    filename = "memory.txt"
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("")
    paragraphs_local_memory = parse_file(filename)
    embeddings = get_embeddings(filename, paragraphs_local_memory)
    threading.Thread(target=audio_worker, daemon=True).start()
    while True:
        prompt = input("<You> ")
        memory_idx = retrieve_memory_context(embeddings, paragraphs_local_memory, prompt)
        research_idx = handle_search_request(prompt)
        bot_response = stream_response(prompt)
        if research_idx >= 0:
            conversation.pop(research_idx)
        if memory_idx >= 0:
            conversation.pop(memory_idx)
        embeddings, paragraphs_local_memory = save_information_from_index(
            prompt, bot_response, filename, embeddings, paragraphs_local_memory
        )

# =====================
# Script Entry Point
# =====================
if __name__ == "__main__":
    asyncio.run(main())