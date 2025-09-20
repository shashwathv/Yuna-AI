import requests
import json
import os
import sys
import threading
import subprocess
import re
import time

# --- Configuration ---
YUNA_API_URL = "http://127.0.0.1:5000/chat"
HEALTH_CHECK_URL = "http://127.0.0.1:5000/health"
HISTORY_FILE = "yuna_chat_history.json"
MAX_HISTORY_TURNS = 5  # Reduced to prevent context issues
VOICE_MODEL_PATH = os.path.expanduser("~/.local/share/piper/voices/en_US/amy/medium/en_US-amy-medium.onnx")

# --- Character Validation ---
def validate_response(response):
    """Check if response maintains character and fix if needed"""
    forbidden_phrases = [
        "as an ai", "i am an ai", "language model", 
        "my training", "i don't have feelings", "my knowledge cutoff"
    ]
    
    response_lower = response.lower()
    for phrase in forbidden_phrases:
        if phrase in response_lower:
            print("\n⚠️ [Character break detected - applying correction]")
            return "*adjusts apron* My apologies Master, I seemed to have lost myself for a moment. How may I serve you?"
    
    # Ensure Master is used
    if "master" not in response_lower and len(response) > 20:
        response += ", Master"
    
    return response

def clean_speech_text(text):
    """Remove all non-speech elements while preserving Yuna's personality markers"""
    # Remove any character correction notices
    text = re.sub(r'\[Character correction applied\]', '', text)
    # Remove XML/SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert actions to spoken descriptions
    text = re.sub(r'\*(.*?)\*', r'', text)  # Remove action descriptions for TTS
    # Remove special characters
    text = text.replace("~", "").replace("💖", "").replace("🌸", "")
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
    """Save conversation history with character validation"""
    try:
        # Validate all responses before saving
        for turn in history:
            if "ai" in turn:
                turn["ai"] = validate_response(turn["ai"])
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Couldn't save history: {e}")

def load_history():
    """Load and validate conversation history"""
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            # Validate loaded responses
            for turn in history:
                if "ai" in turn:
                    turn["ai"] = validate_response(turn["ai"])
            return history
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception as e:
        print(f"⚠️ Couldn't load history: {e}")
        return []

def check_service_health():
    """Check if the Yuna service is running and properly configured"""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("character") == "Yuna Aisaka":
                return True
    except:
        pass
    return False

# --- Initial Greeting ---
def get_greeting():
    """Generate Yuna's greeting"""
    greetings = [
        "*bows politely* Welcome home, Master! How may I serve you today?",
        "*adjusts apron* Good to see you, Master. Shall I prepare some tea?",
        "*curtsies* Master has returned! How may Yuna assist you?",
    ]
    import random
    return random.choice(greetings)

# --- Main Chat Loop ---
def main():
    print("🌸 Yuna Aisaka - Maid Service Client 🌸")
    print("=" * 50)
    
    # Check service health
    if not check_service_health():
        print("⚠️ Warning: Yuna service may not be properly configured")
        print("Please ensure yuna_service.py is running correctly")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    conversation_history = load_history()
    farewell_keywords = ["exit", "quit", "goodbye", "bye", "see you later", "farewell"]
    
    # Initial greeting
    greeting = get_greeting()
    print(f"\nYuna: {greeting}")
    speak(greeting)
    time.sleep(0.5)
    
    print("\n(Type 'exit' or 'quit' to leave)")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
        except KeyboardInterrupt:
            print("\n\nYuna: *bows deeply* It seems Master wishes to leave. Please take care!")
            save_history(conversation_history)
            break

        if user_input.lower().strip() in farewell_keywords:
            save_history(conversation_history)
            farewell_text = "*curtsies* I shall await your return, Master. Please take care!"
            print(f"\nYuna: {farewell_text}")
            speak(farewell_text)
            time.sleep(2)
            break

        # Character reinforcement in user input
        if "ai" in user_input.lower() or "robot" in user_input.lower():
            print("\n[Note: Yuna is a maid character, not an AI]")

        payload = {
            'user_input': user_input,
            'history': conversation_history[-MAX_HISTORY_TURNS:]
        }

        try:
            with requests.post(YUNA_API_URL, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                full_response = ""
                print("Yuna: ", end="", flush=True)
                
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        # Filter out correction notices from display
                        display_chunk = chunk.replace("[Character correction applied]", "")
                        print(display_chunk, end="", flush=True)
                        full_response += display_chunk
                
                print()  # New line after response
                
                # Validate and correct response if needed
                full_response = full_response.strip()
                validated_response = validate_response(full_response)
                
                if validated_response != full_response:
                    print(f"Yuna: {validated_response}")
                    full_response = validated_response
                
                if full_response:
                    speak(full_response)
                    conversation_history.append({
                        "user": user_input, 
                        "ai": full_response
                    })
                    
                    # Auto-save every 3 turns
                    if len(conversation_history) % 3 == 0:
                        save_history(conversation_history)

        except requests.exceptions.Timeout:
            print("\n⏱️ [Response timeout - Yuna seems to be thinking too hard]")
            fallback = "*bows apologetically* Forgive me Master, I need a moment to collect myself."
            print(f"Yuna: {fallback}")
            speak(fallback)
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ [Error: Could not connect to Yuna service]")
            print(f"[Details: {str(e)[:100]}]")
            print("[Please ensure yuna_service.py is running on port 5000]")
            sys.exit(1)

if __name__ == "__main__":
    main()