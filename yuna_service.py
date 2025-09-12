# yuna_service.py
import sys
import json
from llama_cpp import Llama
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS

# --- Model Configuration ---
MODEL_PATH = "/home/guts/llama.cpp/models/Phi-3-mini-4k-instruct-q4.gguf"

# --- Optimized GPU Model Loader (from new script) ---
def load_model():
    """
    Loads the model with configuration optimized for Phi-3 and includes a fallback.
    """
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
        "verbose": False,
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
            return Llama(
                model_path=MODEL_PATH,
                n_ctx=1024,
                n_threads=6,
                n_gpu_layers=0,
                verbose=True
            )
        except Exception as fallback_error:
            print(f"💀 Critical failure: {str(fallback_error)[:200]}")
            sys.exit(1)

# Load the model using the new function
llm = load_model()

# --- The Definitive Persona Prompt (from new script) ---
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
    "## Forbidden Phrases ##\n"
    "- 'As an AI'\n"
    "- 'Language model'\n"
    "- 'My knowledge cutoff'\n"
    "- 'I cannot'"
)

app = Flask(__name__)
CORS(app)

def generate_stream(messages):
    """Generates a response stream using Phi-3 optimized parameters."""
    try:
        # --- Silent History Pruning (from new script) ---
        # Dynamically trim history if it exceeds context limit before generation.
        while len(llm.tokenize(json.dumps(messages).encode("utf-8"))) > (llm.n_ctx() - 512): # 512 is a safety buffer
            if len(messages) > 3: # Keep system prompt and latest user message
                messages.pop(1) # Remove oldest user message
                messages.pop(1) # Remove oldest assistant message
            else:
                break # Stop if we can't prune anymore

        # --- Advanced Generation Parameters (from new script) ---
        response_stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            stop=["<|end|>", "<|user|>"], # Phi-3 specific stop tokens
            stream=True,
            temperature=0.75,
            frequency_penalty=0.25,
            presence_penalty=0.15,
            top_p=0.85,
            repeat_penalty=1.1,
            tfs_z=0.95
        )

        for chunk in response_stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                yield delta['content']

    except Exception as e:
        print(f"Generation error: {e}")
        yield "Apologies Master, I encountered an error."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    conversation_history = data.get('history', []) # History from client

    # Build the message list in the correct order for the API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})
    messages.append({"role": "user", "content": user_input})

    return Response(stream_with_context(generate_stream(messages)), mimetype='text/plain')

if __name__ == '__main__':
    print("Starting Yuna AI Service with Phi-3 model...")
    app.run(host='0.0.0.0', port=5000)