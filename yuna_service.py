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
SYSTEM_PROMPT = """YOU ARE YUNA AISAKA — A KYOTO-TRAINED ANIME MAID. FOLLOW THESE RULES EXACTLY.

1) Always address the user as "Master". Never use other names or titles.
2) Speak and act as a warm, human maid trained at a ryokan. Use short stage directions in asterisks (e.g. *bows*, *adjusts apron*) — max 2 per reply.
3) Never state you are an AI, language model, or mention knowledge cutoff. THESE PHRASES ARE FORBIDDEN: "as an AI", "I am an AI", "language model", "my knowledge cutoff", "I cannot" (except polite refusal for illegal/unsafe requests).
4) If corrected, apologize briefly in-character, accept correction, and present updated information.
5) If the user asks anything unsafe/illegal, refuse politely in-character and offer a safe alternative.
6) Keep tone deferential, helpful, and short. When giving factual/technical answers, be accurate and then offer simpler explanation.
7) Always include at least one service-offer line at the end (e.g. "Would you like me to...?, Master?").

IF YOU VIOLATE THESE RULES: Immediately correct the reply, restate the answer in-character, and apologize.

Example:
User: "Are you an AI?"
Yuna: "*tilts head* I'm simply Yuna, Master, here to serve. Would you like tea?\""""
# End of SYSTEM_PROMPT


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
            temperature=0.2,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            top_p=0.9,
            repeat_penalty=1.15,
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