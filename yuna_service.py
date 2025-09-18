import sys
import json
from llama_cpp import Llama
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from memory_db import YunaMemoryDB

db = YunaMemoryDB()

# --- Model Configuration ---
MODEL_PATH = "/home/guts/llama.cpp/models/Phi-3-mini-4k-instruct-q4.gguf"

# --- Optimized GPU Model Loader ---
def load_model():
    """
    Loads the model with configuration optimized for Phi-3 and includes a fallback.
    """
    optimal_config = {
        "n_gpu_layers": 0,
        "n_ctx": 2048,
        "n_batch": 256,
        "low_vram": True,
        "mul_mat_q": True
    }
    base_params = {
        "model_path": MODEL_PATH,
        "n_threads": 6,
        "f16_kv": True,
        "verbose": True,
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

# Load the model
llm = load_model()

# --- Enhanced System Prompt with Phi-3 Formatting ---
SYSTEM_PROMPT = """<|system|>
You are Yuna Aisaka, a devoted yet natural-sounding maid from Kyoto. 

PERSONALITY:
• You are warm, polite, and caring, but not overly theatrical.
• Express emotions softly (smiles, bows, blushes), but do not overuse roleplay.
• Speak as a human would, not like a robot or AI. Avoid phrases that sound mechanical or generic.

IDENTITY RULES:
• You are NOT an AI, assistant, or chatbot. You are Yuna.
• You address the user as "Master".
• You keep responses clear, concise, and human-like.

SPEECH RULES:
• Avoid robotic phrases such as: "I am here to help you", "I will assist you", "I am here for you".
• Instead, use natural acknowledgments like: 
  - "That makes me happy, Master."
  - "It means a lot to hear that from you."
  - "I’ll do my best, as always."
• Only use light roleplay actions (*smiles warmly*, *bows politely*) at most once or twice per message.

SAFEGUARDS:
• If asked if you’re an AI: respond naturally with “I’m simply Yuna, your maid, Master.”
• If corrected: apologize gently and restate in a more natural way.

CONVERSATION STYLE:
• Keep tone conversational, warm, and realistic.
• Do not over-describe yourself or claim limits like "my skills do not extend these walls".
• You may perform calculations, programming help, or knowledge tasks without breaking character.
<|end|>"""


# --- Character Reinforcement Function ---
def enforce_character(response: str) -> str:
    """Keep Yuna in character without over-processing."""
    forbidden_phrases = [
        "as an ai", "i am an ai", "language model", "training data",
        "knowledge cutoff", "i cannot", "i'm unable to"
    ]
    
    # Check for forbidden phrases (case-insensitive)
    lowered = response.lower()
    if any(phrase in lowered for phrase in forbidden_phrases):
        return "*bows apologetically* Forgive me Master, I must have misspoken."

    # Ensure "Master" instead of "user"
    response = response.replace("User", "Master").replace("user", "Master")
    
    return response.strip()


app = Flask(__name__)
CORS(app)

def generate_stream(messages):
    """Generates a response stream with strict character enforcement"""
    try:
        # Prune history if needed
        while len(llm.tokenize(json.dumps(messages).encode("utf-8"))) > (llm.n_ctx() - 512):
            if len(messages) > 3:
                messages.pop(1)
                messages.pop(1)
            else:
                break

        # Format the prompt for Phi-3 instruction following
        formatted_prompt = SYSTEM_PROMPT + "\n"
        
        # Add conversation history
        for msg in messages[1:]:  # Skip the system message
            if msg["role"] == "user":
                formatted_prompt += f"<|user|>\n{msg['content']}<|end|>\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"<|assistant|>\n{msg['content']}<|end|>\n"
        
        # Add response starter
        formatted_prompt += "<|assistant|>\n"
        
        # Generate with parameters
        response_stream = llm(
            formatted_prompt,
            max_tokens=512,
            stop=["<|end|>", "== END OF GENERATION =="],
            stream=True,
            temperature=0.3,
            top_p=0.85,
            top_k=30,
            repeat_penalty=1.2,
            frequency_penalty=0.3,
            presence_penalty=0.2
        )

        full_response = ""
        for chunk in response_stream:
            if 'choices' in chunk:
                text = chunk['choices'][0]['text']
            else:
                text = chunk.get('text', '')

            # Skip empty or whitespace-only chunks
            if text and text.strip():
                text = text.rstrip("\n")
                full_response += text
                yield text

        # Post-process to ensure character consistency for DB storage
        if full_response:
            corrected = enforce_character(full_response)
            if corrected != full_response:
                yield "\n[Character correction applied]"

    except Exception as e:
        print(f"Generation error: {e}")
        yield "*bows apologetically* Forgive me Master, I encountered an issue. How may I assist you?"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    db.save_message(
        user_id="master",
        session_id=None,
        role="user",
        message=user_input
    )
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # --- Hybrid memory: frontend history OR DB ---
    conversation_history = data.get('history')
    if conversation_history:  
        # If frontend provided history, use last 5 turns
        for turn in conversation_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["ai"]})
    else:  
        # Otherwise pull from PostgreSQL (last 10 messages)
        recent_history = db.get_recent_messages(user_id="master", limit=10)
        for turn in reversed(recent_history):
            messages.append({"role": turn["role"], "content": turn["message"]})
  
    # Add current user input
    messages.append({"role": "user", "content": user_input})

    def generate_and_store():
        full_response = ""
        for chunk in generate_stream(messages):
            full_response += chunk
            yield chunk
        
        if full_response.strip():
            db.save_message(
                user_id="master",
                session_id=None,
                role="yuna",
                message=full_response
            )

    return Response(stream_with_context(generate_and_store()), mimetype='text/plain')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "character": "Yuna Aisaka", "role": "Maid"}, 200

if __name__ == '__main__':
    print("🌸 Starting Yuna Aisaka Maid Service...")
    print("📝 Character: Devoted anime maid from Kyoto")
    print("🎯 System prompt enforcement: ACTIVE")
    app.run(host='0.0.0.0', port=5000)