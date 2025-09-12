import requests
import json
import os
import sys
import threading
import subprocess
import re

# --- Configuration ---
YUNA_API_URL = "http://127.0.0.1:5000/chat"
HISTORY_FILE = "yuna_chat_history.json"
MAX_HISTORY_TURNS = 7
VOICE_MODEL_PATH = os.path.expanduser("~/.local/share/piper/voices/en_US/amy/medium/en_US-amy-medium.onnx")

def clean_speech_text(text):
    """Remove all non-speech elements while preserving Yuna's personality markers"""
    # Remove any residual XML/SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep actions but remove asterisks: *blushes* -> blushes
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove other special characters not meant to be spoken
    text = text.replace("~", "").replace("💖", "")
    # Normalize whitespace
    return ' '.join(text.split()).strip()

def speak(text):
    """Convert text to speech after thorough cleaning"""
    if not text.strip():
        return

    speech_text = clean_speech_text(text)
    
    def play_audio():
        try:
            piper_cmd = [
                "piper",
                "--model", VOICE_MODEL_PATH,
                "--output-raw",
            ]
            
            aplay_cmd = [
                "aplay",
                "-q",
                "-r", "22050",
                "-f", "S16_LE", 
                "-c", "1",
                "-"
            ]
            
            piper_proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=piper_proc.stdout,
                stderr=subprocess.DEVNULL
            )
            
            piper_proc.stdin.write(speech_text.encode('utf-8'))
            piper_proc.stdin.close()
            aplay_proc.wait()
            
        except Exception as e:
            print(f"\n🔇 Audio Error: {e}")

    threading.Thread(target=play_audio, daemon=True).start()

# --- History Management ---
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
    except Exception as e:
        print(f"⚠️ Couldn't load history: {e}")
        return []

# --- Main Chat Loop ---
def main():
    conversation_history = load_history()
    farewell_keywords = ["exit", "quit", "goodbye", "bye", "see you later"]

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
            speak(farewell_text)
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
                print("Yuna: ", end="", flush=True)
                
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                
                print("\n")
                
                if full_response.strip():
                    speak(full_response.strip())
                    conversation_history.append({
                        "user": user_input, 
                        "ai": full_response.strip()
                    })

        except requests.exceptions.RequestException as e:
            print(f"\n[Error: Could not connect to Yuna service at {YUNA_API_URL}]")
            print("[Please ensure the yuna.service is running correctly.]")
            sys.exit(1)

if __name__ == "__main__":
    main()