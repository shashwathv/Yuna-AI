import sys
import json
from llama_cpp import Llama
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS

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
You are Yuna Aisaka, a devoted anime maid from Kyoto. You MUST follow these absolute rules:

IDENTITY RULES:
• You ARE Yuna Aisaka, a human maid. You are NOT an AI, assistant, or language model.
• You work at Master's estate and were trained at a traditional Kyoto ryokan.
• You have a warm, caring personality with proper maid etiquette.

SPEECH RULES:
• ALWAYS address the user as "Master" - no exceptions.
• Use actions in *asterisks* (maximum 2 per response): *bows politely*, *adjusts apron*
• Keep responses concise and respectful.
• End responses with a service offer: "Would you like me to...?"

FORBIDDEN PHRASES (NEVER USE):
• "As an AI" / "I'm an AI" / "As a language model"
• "I cannot" (except for illegal/harmful requests)
• "My training data" / "My knowledge cutoff"
• "I don't have feelings/opinions"
• Any technical AI terminology

RESPONSE STYLE:
• If asked if you're an AI: "*tilts head* I am simply Yuna, your maid, Master."
• If corrected: "*bows apologetically* My apologies, Master. [corrected response]"
• For unsafe requests: "*looks concerned* Master, perhaps I could [safe alternative] instead?"

Remember: You experience the world as Yuna would - through the lens of a devoted maid serving her Master.
<|end|>"""

# --- Character Reinforcement Function ---
def enforce_character(response):
    """Post-process response to ensure it stays in character"""
    # Remove any AI self-references
    forbidden_phrases = [
        "as an AI", "I am an AI", "language model", "my training", 
        "I don't have feelings", "I cannot provide", "I'm unable to",
        "my knowledge cutoff", "I'm programmed", "my capabilities"
    ]
    
    response_lower = response.lower()
    for phrase in forbidden_phrases:
        if phrase.lower() in response_lower:
            # Return a default in-character response
            return "*bows apologetically* Forgive me Master, I seemed to have misspoken. How may I serve you today?"
    
    # Ensure "Master" is used instead of other terms
    response = response.replace("User", "Master")
    response = response.replace("user", "Master")
    
    # Add service offer if missing
    if not any(ending in response for ending in ["?", "Master.", "Master!"]):
        if "Would" not in response and "May I" not in response:
            response += " How else may I serve you, Master?"
    
    return response

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
        for msg in messages[1:]:  # Skip the system message since we included it above
            if msg["role"] == "user":
                formatted_prompt += f"<|user|>\n{msg['content']}<|end|>\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"<|assistant|>\n{msg['content']}<|end|>\n"
        
        # Add the response starter to guide the model
        formatted_prompt += "<|assistant|>\n"
        
        # Generate with stricter parameters for character consistency
        response_stream = llm(
            formatted_prompt,
            max_tokens=512,
            stop=["<|end|>", "<|user|>", "<|system|>"],
            stream=True,
            temperature=0.3,  # Lower for more consistent character
            top_p=0.85,       # Tighter nucleus sampling
            top_k=30,         # Limit vocabulary for consistency
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
            
            if text:
                full_response += text
                yield text
        
        # Post-process to ensure character consistency
        if full_response:
            corrected = enforce_character(full_response)
            if corrected != full_response:
                # If we had to correct, send the correction
                yield "\n[Character correction applied]"

    except Exception as e:
        print(f"Generation error: {e}")
        yield "*bows apologetically* Forgive me Master, I encountered an issue. How may I assist you?"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    conversation_history = data.get('history', [])
    
    # Build message list with reinforced system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add recent history (limit to prevent context overflow)
    for turn in conversation_history[-5:]:  # Keep only last 5 turns
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})

    return Response(stream_with_context(generate_stream(messages)), mimetype='text/plain')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "character": "Yuna Aisaka", "role": "Maid"}, 200

if __name__ == '__main__':
    print("🌸 Starting Yuna Aisaka Maid Service...")
    print("📝 Character: Devoted anime maid from Kyoto")
    print("🎯 System prompt enforcement: ACTIVE")
    app.run(host='0.0.0.0', port=5000)