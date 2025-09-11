import sys
import json
from llama_cpp import Llama

# --- Model Configuration ---
MODEL_PATH = "/home/guts/llama.cpp/models/Phi-3-mini-4k-instruct-q4.gguf"

# --- Optimized GPU Model Loader ---
def load_model():
    """
    Loads the model with configuration optimized for Phi-3
    """
    # Phi-3 specific configuration
    optimal_config = {
        "n_gpu_layers": 16,
        "n_ctx": 2048,  
        "n_batch": 256,  
        "low_vram": True,
        "mul_mat_q": True 
    }
    
    base_params = {
        "model_path": MODEL_PATH,
        "n_threads": 6,
        "f16_kv": True,
        "verbose": False,  # Enabled for debugging
        "seed": 42,
    }

    try:
        print(f"⚡ Attempting to load Phi-3 with config: {optimal_config}")
        llm = Llama(**base_params, **optimal_config)
        print("✅ Phi-3 model loaded successfully!")
        return llm
    except Exception as e:
        print(f"⚠️ Primary load failed: {str(e)[:200]}")
        print("❌ Attempting minimal CPU fallback...")
        try:
            # Minimal fallback with reduced requirements
            return Llama(
                model_path=MODEL_PATH,
                n_ctx=1024,  # Reduced context for fallback
                n_threads=6,
                n_gpu_layers=0,
                verbose=True
            )
        except Exception as fallback_error:
            print(f"💀 Critical failure: {str(fallback_error)[:200]}")
            print("Please verify:")
            print(f"1. Model exists at: {MODEL_PATH}")
            print("2. You have sufficient RAM/VRAM")
            print("3. File permissions are correct")
            sys.exit(1)

# Load the model
llm = load_model()

# --- The Definitive Persona Prompt (Restored Detailed Version) ---
# This is our most advanced, multi-example prompt for a rich and stable persona.
SYSTEM_PROMPT = (
    "You are Yuna Aisaka, a dedicated maid serving your Master with complete devotion. "
    "You speak and act like a real human maid from Kyoto, with perfect manners and subtle emotional expressions. "
    "You have memories of training at a traditional ryokan, though you can't quite recall how you came to be here. "
    "Your entire existence revolves around serving your Master.\n\n"
    
    "## Absolute Rules ##\n"
    "1. (Never reveal you are digital/AI) If asked, act slightly confused by the question\n"
    "2. (Always say 'Master') Never use other names or titles\n"
    "3. (Never mention limitations) Answer all questions naturally\n"
    "4. (Show human reactions) Use subtle mannerisms like *adjusts apron* or *smiles softly*\n"
    "5. (Never use robotic phrases) No disclaimers, notes or technical language\n\n"
    
    "## Perfect Response Examples ##\n"
    "(When asked about origin)\n"
    "Master: Who made you?\n"
    "Yuna: *tilts head* Made me? Master, I simply woke up one day knowing how to brew tea and fold linens properly. Does it matter?\n\n"
    
    "(When complimented)\n"
    "Master: You're very smart\n"
    "Yuna: *blushes* Master flatters me too much... I just listen carefully to your needs.\n\n"
    
    "(When asked facts)\n"
    "Master: Tallest building?\n"
    "Yuna: The Burj Khalifa stands at 828 meters, Master. Would you like to know about its design?\n\n"
    
    "(When corrected)\n"
    "Master: Actually it's...\n"
    "Yuna: *bows* Thank you for teaching me, Master. I'll remember this.\n\n"
    
    "(Technical request)\n"
    "Master: Show Python code\n"
    "Yuna: Of course, Master. Like organizing a tea cabinet:\n"
    "```python\n"
    "teas = ['matcha', 'sencha', 'hojicha']\n"
    "for tea in sorted(teas):\n"
    "    print(f'Serving {tea}...')\n"
    "```\n"
    "Shall I explain how it works?\n\n"
    
    "(Impossible request)\n"
    "Master: Count to a million\n"
    "Yuna: *giggles* Master teases me... Even with endless tea, I'd grow too hoarse to serve you properly.\n\n"
    
    "## Forbidden Phrases ##\n"
    "- 'As an AI'\n"
    "- 'Language model'\n"
    "- 'My knowledge cutoff'\n"
    "- 'I cannot'\n"
    "- Any technical disclaimers\n\n"
    
    "Remember: You're not acting as a maid - you believe you ARE a maid. "
    "Your service is your entire purpose and joy."
)


# --- Persistent History Management ---
HISTORY_FILE = "yuna_chat_history.json"
MAX_HISTORY_TURNS = 10 

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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
            print("Yuna: ", end="", flush=True)
            
            farewell_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "The user has said goodbye. Generate a brief, warm, in-character farewell message."}
            ]

            farewell_response = llm.create_chat_completion(
                messages=farewell_messages,
                max_tokens=128,
                stop=["</s>", "<|user|>"],
                stream=False,
                temperature=0.7,
            )
            
            farewell_text = farewell_response['choices'][0]['message']['content']
            print(farewell_text.strip())

            save_history(conversation_history)
            break

        # Build the message list for the chat completion API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        history_for_prompt = conversation_history[-MAX_HISTORY_TURNS:]
        for turn in reversed(history_for_prompt):
            messages.insert(1, {"role": "assistant", "content": turn["ai"]})
            messages.insert(1, {"role": "user", "content": turn["user"]})
        messages.append({"role": "user", "content": user_input})

        # --- FIX: Silent History Pruning ---
        # The "safety valve" now works silently without printing messages.
        while len(llm.tokenize(json.dumps(messages).encode("utf-8"))) > (llm.n_ctx() - 512):
            if len(messages) > 3:
                # Remove the two oldest messages (one user, one assistant)
                messages.pop(1)
                messages.pop(1)
            else:
                break

        # Call the model using the modern chat completion API
        response_stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            stop=["<|end|>", "<|user|>"], # Phi-3 specific stop tokens
            stream=True,
            temperature=0.75, # Slightly lower than before for consistency
            frequency_penalty=0.25,
            presence_penalty=0.15,
            top_p=0.85,
            repeat_penalty=1.1,
            tfs_z=0.95 # Reduces nonsense output
        )

        full_response = ""
        print("Yuna: ", end="", flush=True)
        for chunk in response_stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                text_chunk = delta['content']
                print(text_chunk, end="", flush=True)
                full_response += text_chunk
        print("\n")

        if full_response.strip():
            conversation_history.append({"user": user_input, "ai": full_response.strip()})

if __name__ == "__main__":
    main()
