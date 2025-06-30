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
import sys
import shutil
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
from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd
import time
import sys

# =====================
# Global Variables & Constants
# =====================
MODEL_CHATBOT = "gemma3n:e4b"  # Chatbot model name
MODEL_EMB = "nomic-embed-text"  # Embedding model name
MODEL_VOICE_TRASC = "large-v3"


SYSTEM_LANGUAGE = "italian"  # System language for translation
VOICE = "if_sara"  # Default TTS voice
VOICE_TOGGLED = True   # Whether voice output is enabled


audio_queue = queue.Queue()  # Queue for audio playback requests
generated_audio_queue = queue.Queue()  # New queue for pre-generated audio data
conversation = []  # History of conversation messages

TRANSLATION_CACHE_FILE = "translations.json"
translation_cache = {}

def load_translation_cache():
    global translation_cache
    if os.path.exists(TRANSLATION_CACHE_FILE):
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
        except Exception:
            translation_cache = {}
    else:
        translation_cache = {}

def save_translation_cache():
    with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(translation_cache, f, ensure_ascii=False, indent=2)

# =====================
# Utility Functions
# =====================
def auto_translate(text):
    """
    Translates text into the system language (if not English).
    Caches translations to avoid redundant requests.
    """
    if SYSTEM_LANGUAGE.lower() == 'english':
        return text
    cache_key = f"{SYSTEM_LANGUAGE.lower()}::{text}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
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
    translated = response["message"]["content"]
    translation_cache[cache_key] = translated
    save_translation_cache()
    return translated

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
        if (content.endswith(('.', '?', '!', ':')) and len(tmp_content) > 30):
            play_audio(tmp_content)
            tmp_content = ''
        response += content
        print(content, end='', flush=True)
    if len(tmp_content) > 1:
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
            print(f"<memory> {paragraphs[index]}")
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
            ("You are a memory generator. Given the full conversation, extract data that the assistant must remember."
             "Respond with one or more concise sentences (each on a new line).")
        )
    }
    temp_conversation = [system_prompt, {"role": "user", "content": f"user: {prompt}\nAI: {response_ai}"}]
    response = ollama.chat(model=MODEL_CHATBOT, messages=temp_conversation, options={"temperature": 0}, stream=False)
    response = response["message"]["content"]
    memories = response.split('\n')
    if memories == [""]: 
        return embeddings, paragraphs
    # add datatime to memories
    datatime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memories = [ datatime_now+' - '+m for m in memories if m.strip()  ]
        
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
    # remove '*' from text
    text = text.replace('*', '')
    if VOICE_TOGGLED:
        audio_queue.put(text)

def audio_generator():
    """
    Pre-generates audio samples concurrently from queued text.
    """
    kokoro = Kokoro("voice/kokoro-v1.0.onnx", "voice/voices-v1.0.bin")
    while True:
        text = audio_queue.get()
        g2p = EspeakG2P(language=SYSTEM_LANGUAGE[:2])
        phonemes, _ = g2p(text)
        samples, sample_rate = kokoro.create(phonemes, VOICE, is_phonemes=True)
        num_channels = samples.shape[1] if samples.ndim > 1 else 1
        bytes_per_sample = samples.dtype.itemsize
        generated_audio_queue.put((samples, sample_rate, num_channels, bytes_per_sample))
        audio_queue.task_done()

def audio_worker():
    """
    Plays pre-generated audio samples as soon as available.
    """
    while True:
        samples, sample_rate, num_channels, bytes_per_sample = generated_audio_queue.get()
        wave_obj = sa.WaveObject(samples.tobytes(), num_channels, bytes_per_sample, sample_rate)
        wave_obj.play().wait_done()
        generated_audio_queue.task_done()

def record_until_silence( silence_threshold=0.2, silence_duration=2, sample_rate=44100):
    """
    Registra l'audio finché non viene rilevato silenzio prolungato, poi salva in MP3.
    """
    # Attendi che il TTS abbia terminato la riproduzione audio
    audio_queue.join()
    generated_audio_queue.join()
    channels = 1
    buffer = []
    last_sound_time = time.time()

   
    def callback(indata, frames, time_info, status):
        nonlocal last_sound_time
        volume_norm = np.sqrt(np.mean(indata**2))
        buffer.append(indata.copy())
        if volume_norm > silence_threshold:
            last_sound_time = time.time()

    
    with sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate):
        max_dots = shutil.get_terminal_size().columns - len("<You> ") - 1
        if max_dots < 2:
            max_dots = 1
        while True:
            time_of_silence = time.time() - last_sound_time
            if time_of_silence > silence_duration :
                # Clear the printed line before exiting
                sys.stdout.write("\r" + " " * (len("<You> ") + max_dots) + "\r")
                sys.stdout.flush()
                break
            
            ratio = time_of_silence / silence_duration
            dot_count = int((1 - ratio) * max_dots)
            # Assicuriamoci di avere almeno 1 punto
            if dot_count < 1:
                dot_count = 1
            dots = "." * dot_count
            sys.stdout.write("\r<You> " + dots + " " * (max_dots - dot_count))
            sys.stdout.flush()
            time.sleep(0.1)
    if not buffer:
        return False
    audio_data = np.concatenate(buffer, axis=0)
    temp_wav = "voice/tmp.wav"
    sf.write(temp_wav, audio_data, sample_rate)
    return True
    
def get_input_from_microphone(model, silence_threshold):
    
    while not record_until_silence(silence_threshold = silence_threshold, silence_duration=2, sample_rate=44100):
        time.sleep(0.1)
        
    print("\r<You> ", end='', flush=True)
    segments, _ = model.transcribe("voice/tmp.wav", word_timestamps=True,  vad_filter=True)
    output = ''
    for segment in segments:
        if segment.words:
            for word in segment.words:
                output += word.word + ' '
                print(f"{word.word}", end='', flush=True)
    if output != '':
        print("")
    return output.strip()

def calib_silence_threshold(duration=5, channels=1, sample_rate=44100):
    """
    Calibrates the silence threshold by recording ambient noise for a given duration.
    Returns an appropriate silence threshold value.
    """
    print(auto_translate("Calibrating silence threshold. Please remain quiet..."))
    recordings = []
    def callback(indata, frames, time_info, status):
        recordings.append(indata.copy())

    with sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate):
        # Record for the specified duration in milliseconds
        sd.sleep(int(duration * 1000))

    if not recordings:
        print("No data recorded during calibration.")
        return 0.2  # Fallback threshold

    audio_data = np.concatenate(recordings, axis=0)
    # Compute RMS for each frame
    frame_size = 1024
    rms_values = []
    for start in range(0, len(audio_data), frame_size):
        frame = audio_data[start:start+frame_size]
        if frame.size == 0:
            continue
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    mean_rms = np.mean(rms_values)
    threshold = mean_rms * 1.8
    print(f"Calibrated silence threshold: {threshold:.4f}")
    os.system('cls' if os.name == 'nt' else 'clear')
    return threshold

# =====================
# Main Application Entry Point
# =====================
async def main():
    """
    Main loop: Handles user input while retrieving context, search results, and processing responses.
    """
    load_translation_cache()
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
    # Start audio generation and playback threads
    threading.Thread(target=audio_generator, daemon=True).start()
    threading.Thread(target=audio_worker, daemon=True).start()

    model = WhisperModel(MODEL_VOICE_TRASC, device="cpu", compute_type="int8")
    silence_threshold = calib_silence_threshold() 
    while True:
        prompt = get_input_from_microphone(model, silence_threshold = silence_threshold)
        if not prompt:
            continue
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