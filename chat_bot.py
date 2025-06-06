# =====================
# Imports and Constants
# =====================
import numpy as np
import ollama
import os
import json
from PIL import Image, ImageGrab
from numpy.linalg import norm
from io import BytesIO
import base64
import asyncio
import hashlib
import soundfile as sf  
from misaki import espeak 
from misaki.espeak import EspeakG2P 
import simpleaudio as sa
from kokoro_onnx import Kokoro 
import agent_web
import re
import threading
import queue

# =====================
# Global Variables
# =====================


MODEL_CHATBOT = "gemma3:4b"  # Name of the chatbot model to use
MODEL_EMB = "nomic-embed-text"  # Name of the model for embeddings

SYSTEM_LANGUAGE = "italian"  # System language for translation
VOICE = "if_sara"  # Default voice for the assistant
VOICE_TOGGLED = True  # Enable/disable voice output



audio_queue = queue.Queue()  # Queue for audio playback requests
conversation = []  # Stores the conversation history as a list of message dicts

# =====================
# Utility Functions
# =====================

def auto_translate(text):
    """
    Translate the given text into the system language using the chatbot model.
    If the system language is English, return the text unchanged.
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
    Read a file and split its content into paragraphs (separated by blank lines).
    Returns a list of paragraphs.
    """
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    """
    Save embeddings to a JSON file in the 'embeddings' directory.
    """
    name = filename.split(".")[0]
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{name}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    """
    Load embeddings from a JSON file if it exists, otherwise return False.
    """
    path = f"embeddings/{filename}.json"
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        return json.load(f)


def get_embeddings(filename, chunks):
    """
    Generate or load embeddings for a list of text chunks (paragraphs).
    Uses a hash to avoid recomputation if the file hasn't changed.
    """
    current_hash = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
    loaded = load_embeddings(filename)
    if loaded is not False and loaded.get("hash") == current_hash:
        return loaded["embeddings"]
    embeddings = [
        ollama.embeddings(model=MODEL_EMB, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    data_to_save = {
        "hash": current_hash,
        "embeddings": embeddings
    }
    save_embeddings(filename, data_to_save)
    return embeddings


def find_similar(needle, haystack):
    """
    Find the most similar vectors in haystack to the needle vector using cosine similarity.
    Returns a sorted list of (score, index) tuples.
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# =====================
# Core Chatbot Functions
# =====================

def stream_response(prompt):
    """
    Send the user's prompt to the chatbot and stream the response.
    Plays audio for each sentence and prints the response in real time.
    """
    conversation.append({"role": "user", "content": prompt})
    response = ''
    stream = ollama.chat(model=MODEL_CHATBOT, messages=conversation, stream=True)
    print("<Gemma> ", end='', flush=True)
    # Remove emojis from the response
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    tmp_content = ''
    for chunk in stream:
        content = chunk["message"]["content"]
        content = emoji_pattern.sub("", content)
        tmp_content += content
        # Play audio at sentence boundaries
        if content.endswith(('.', '?', '!', ':')):
            play_audio(tmp_content)
            tmp_content = ''
        response += content
        print(content, end='', flush=True)
    conversation.append({"role": "assistant", "content": response})
    print('')
    return response


def handle_screenshot_request(prompt):
    """
    Decide if a screenshot is needed for the current prompt.
    If so, take a screenshot and add it to the conversation.
    Returns the index of the screenshot message or -1 if not needed.
    """
    conversation_tmp = [
        {
            "role": "system",
            "content": auto_translate(
                "You are not an AI assistant. Your only task is to decide if the last user prompt in a conversation with an AI assistant requires data to be retrieved from computer screen for the assistant to respond correctly. The conversation may or may not already have exactly the context data needed. If the assistant should take a screen shot for more data before responding to ensure a correct response, simply respond \"True\". If the conversation already has the context respond \"False\". Do not generate any explanations. Only generate \"True\" or \"False\" as a response in this conversation using the logic in these instructions.")
        },
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=MODEL_CHATBOT, messages=conversation_tmp, options={"temperature": 0})
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
    Retrieve relevant memory context for the prompt and add it to the conversation.
    Returns the index of the memory message or -1 if no relevant context found.
    """
    prompt_embedding = ollama.embeddings(model=MODEL_EMB, prompt=prompt)["embedding"]
    most_similar_chunks = find_similar(prompt_embedding, embeddings)[:15]
    mem_prompt = "\nContext: "
    for score, index in most_similar_chunks:
        if score > 0.5:
            mem_prompt += paragraphs[index] + "\n"
    if mem_prompt != "\nContext: ":
        conversation.append({"role": "system", "content": mem_prompt})
        return len(conversation) - 1
    return -1


def save_information_from_index(prompt, response_ai, filename, embeddings, paragraphs):
    """
    Analyze the conversation and save new memories if needed.
    Updates embeddings and paragraphs with new memory entries.
    """
    systm_prom = {
        "role": "system",
        "content": auto_translate(
            "You are not an AI assistant that responds to a user. You are an AI memory generator model. You will be given a full conversation to an AI assistant with a User. If you are being used, an AI has determined there are memory that need to be remember. You must determine what the data is the assistant needs to remember. ONLY THE ASSISTENT needs to remember. Do not answer except with one or more short sentences explaining what you have to remember. each sentence goes on a new line.")
    }
    conversation_tmp = [systm_prom, {"role": "user", "content": f"user: {prompt}\nAI: {response_ai}"}]
    response = ollama.chat(model=MODEL_CHATBOT, messages=conversation_tmp, options={"temperature": 0}, stream=False)
    response = response["message"]["content"]
    memories = response.split('\n')
    if memories == [""]:
        return embeddings, paragraphs
    candidate_embeddings = [
        ollama.embeddings(model=MODEL_EMB, prompt=m)["embedding"]
        for m in memories if m.strip()
    ]
    new_embeddings = []
    threshold = 0.8
    with open(filename, "a", encoding="utf-8") as f:
        for mem_text, candidate in zip(memories, candidate_embeddings):
            similarities = [np.dot(candidate, existing) / (norm(candidate) * norm(existing)) for existing in embeddings]
            max_similarity = max(similarities, default=0)
            if max_similarity > threshold:
                continue
            new_embeddings.append(candidate)
            f.write("\n\n" + mem_text)
            paragraphs.append(mem_text)
    embeddings.extend(new_embeddings)
    return embeddings, paragraphs


def handle_search_request(prompt):
    """
    Decide if an internet search is needed for the current prompt.
    If so, ask the user for confirmation and perform the search.
    Returns the index of the search message or -1 if not needed.
    """
    conversation_tmp = [
        {
            "role": "system",
            "content": auto_translate(
                "You are not an AI assistant. Your only task is to decide if the last user prompt in a conversation with an AI assistant requires data to be retrieved from internet. The conversation may or may not already have exactly the context data needed. If the assistant must do an internet search, simply respond \"True\". If the conversation already has the context respond \"False\". Do not generate any explanations. Only generate \"True\" or \"False\" as a response in this conversation using the logic in these instructions. Generate False ")
        },
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=MODEL_CHATBOT, messages=conversation_tmp, options={"temperature": 0})
    response = response["message"]["content"]
    if "true" in response.lower():
        gemma_fixed_response = auto_translate("I can look it up on the internet to answer but it will take a long time, shall I continue??")
        play_audio(gemma_fixed_response)
        request_search = input(f"<Gemma> {gemma_fixed_response} (y/n)\n<You> ").strip().lower()
        if "y" in request_search:
            gemma_fixed_response = auto_translate("I perform the Internet search...")
            print(f"<Gemma> {gemma_fixed_response}")
            play_audio(gemma_fixed_response)
            response = agent_web.run_agent(prompt, step=1)
            conversation.append({
                "role": "assistant",
                "content": auto_translate("I did an online search to find the user's answer, here are the results:") + "\n" + auto_translate(str(response)) + "\n" + auto_translate("Now I can answer the user's question ") + "\n" + prompt
            })
        return len(conversation) - 1
    return -1

# =====================
# Audio Playback Functions
# =====================

def play_audio(text):
    """
    Enqueue text for audio playback if voice is enabled.
    """
    if VOICE_TOGGLED:
        audio_queue.put(text)


def audio_worker():
    """
    Background worker that processes audio playback requests from the queue.
    Uses Kokoro and Misaki G2P for TTS.
    """
    kokoro = Kokoro("voice/kokoro-v1.0.onnx", "voice/voices-v1.0.bin")
    while True:
        text = audio_queue.get()
        g2p = EspeakG2P(language="it")
        phonemes, _ = g2p(text)
        samples, sample_rate = kokoro.create(phonemes, VOICE, is_phonemes=True)
        sf.write("voice/tmp.wav", samples, sample_rate)
        wave_obj = sa.WaveObject.from_wave_file("voice/tmp.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        audio_queue.task_done()

# =====================
# Main Application Entry Point
# =====================

async def main():
    """
    Main application loop. Handles user input, memory, search, and response.
    """
    SYSTEM_PROMPT = auto_translate(
        "You are Gemma, an AI assistant that has another AI model working to get you live data. you can get in from search engine, from you memory or from a a screen shot of the user's computer screen. Results that will be attached before a USER PROMPT. You must analyze it and use any relevant data to generate the most useful & intelligent response an AI assistant that always impresses the user would generate. be breif and concise. you can speak. Use a lot of punctuation to make the answer more expressive. dont use emojis."
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