# chat_client.py (with Streaming TTS)
import requests
import json
import os
import sys
import threading
import subprocess
import re
import queue

# --- Configuration ---
YUNA_API_URL = "http://127.0.0.1:5000/chat"
HISTORY_FILE = "yuna_chat_history.json"
MAX_HISTORY_TURNS = 10
VOICE_MODEL_PATH = os.path.expanduser("~/.local/share/piper/voices/en_US/amy/medium/en_US-amy-medium.onnx")

def clean_speech_text(text):
    """Remove all non-speech elements while preserving Yuna's personality markers"""
    # Remove markdown code blocks for speech
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove asterisks around actions but keep the action text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove other non-spoken characters
    text = text.replace("~", "").replace("💖", "")
    return ' '.join(text.split()).strip()

def tts_worker(tts_queue):
    """
    A dedicated thread that waits for sentences and speaks them.
    Manages a single, persistent piper process.
    """
    try:
        piper_cmd = ["piper", "--model", VOICE_MODEL_PATH, "--output-raw"]
        aplay_cmd = ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"]

        try:
            piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if piper_proc.stdin is None:
                raise RuntimeError("Failed to open pipe to piper process")
            aplay_proc = subprocess.Popen(aplay_cmd, stdin=piper_proc.stdout, stderr=subprocess.DEVNULL)

        except Exception as e:
            print(f"\n🔇 Failed to start audio processes: {e}")
            return

        while True:
            sentence = tts_queue.get()
            if sentence is None:  # Sentinel value to signal exit
                break
            
            speech_text = clean_speech_text(sentence)
            if speech_text:
                piper_proc.stdin.write((speech_text + "\n").encode('utf-8'))
                piper_proc.stdin.flush()
        
        # Cleanly close the processes
        piper_proc.stdin.close()
        piper_proc.wait()
        aplay_proc.wait()

    except FileNotFoundError:
        print("\n🔇 Audio Error: 'piper' or 'aplay' not found. Please ensure they are installed and in your PATH.")
    except Exception as e:
        print(f"\n🔇 Audio Error in worker thread: {e}")

# --- History Management (no changes) ---
def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Couldn't save history: {e}")

def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return []

# --- Main Chat Loop ---
def main():
    conversation_history = load_history()
    farewell_keywords = ["exit", "quit", "goodbye", "bye", "see you later"]

    # --- Setup TTS Worker Thread ---
    sentence_queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,), daemon=True)
    tts_thread.start()

    print("Yuna is ready to serve you, Master 💖 (type 'exit' or 'quit' to leave)")
    while True:
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nYuna: It seems you wish to leave. Farewell for now, Master.")
            save_history(conversation_history)
            break

        if user_input.lower().strip() in farewell_keywords:
            save_history(conversation_history)
            farewell_text = "I shall await your return, Master. Please take care."
            print(f"Yuna: {farewell_text}")
            sentence_queue.put(farewell_text) # Speak the farewell
            threading.Event().wait(3)
            break

        payload = {
            'user_input': user_input,
            'history': conversation_history[-MAX_HISTORY_TURNS:]
        }

        try:
            with requests.post(YUNA_API_URL, json=payload, stream=True) as response:
                response.raise_for_status()

                full_response = ""
                sentence_buffer = ""
                print("Yuna: ", end="", flush=True)

                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                        sentence_buffer += chunk

                        # Check for sentence endings to send to TTS
                        while re.search(r'[.!?]\s', sentence_buffer):
                            sentence_match = re.match(r'(.*?([.!?]\s))', sentence_buffer, re.DOTALL)
                            if sentence_match:
                                sentence = sentence_match.group(1).strip()
                                sentence_queue.put(sentence)
                                sentence_buffer = sentence_buffer[len(sentence):].lstrip()
                            else:
                                break
                
                # After the stream ends, send any remaining text in the buffer
                if sentence_buffer.strip():
                    sentence_queue.put(sentence_buffer.strip())

                print("\n")

                if full_response.strip():
                    conversation_history.append({
                        "user": user_input,
                        "ai": full_response.strip()
                    })

        except requests.exceptions.RequestException as e:
            print(f"\n[Error: Could not connect to Yuna service at {YUNA_API_URL}]")
            break # Exit if the server connection is lost

    # --- Signal TTS worker to exit and cleanup ---
    sentence_queue.put(None)
    tts_thread.join(timeout=5) # Wait for the thread to finish

if __name__ == "__main__":
    main()