import sys
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import torch

# Ensure the local project directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Print system GPU diagnostic info on boot
print("--- Hardware Diagnostics ---")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
print("----------------------------\n")

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

# Initialize FastAPI App
app = FastAPI(title="Otium Code NanoChat HTTPS Agent")

# Configure CORS for your public domain and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://otiumcode.com",
        "https://www.otiumcode.com",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

# -----------------------------------------------------------------------------
# Global LLM Model Initialization
# -----------------------------------------------------------------------------
print("Initializing NanoChat Model and Engine...")
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Load SFT model checkpoint from disk
model, tokenizer, meta = load_model("sft", device, phase="eval")

# Extract special token IDs required for chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Instanimating the generation engine
engine = Engine(model, tokenizer)

# Maintain in-memory conversation history tokens across turns
conversation_tokens = [bos]

print(f"✅ NanoChat loaded successfully on {device}!\n")

# -----------------------------------------------------------------------------
# Streaming Token Generator
# -----------------------------------------------------------------------------
async def generate_llm_stream(user_input: str):
    global conversation_tokens
    
    clean_input = user_input.strip()

    # Handle history reset command
    if clean_input.lower() == "clear":
        conversation_tokens = [bos]
        yield "Conversation history cleared."
        return

    if not clean_input:
        return

    # Add User message tokens to context
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(clean_input))
    conversation_tokens.append(user_end)

    # Prompt Assistant generation sequence
    conversation_tokens.append(assistant_start)

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 512,
        "temperature": 0.6,
        "top_k": 50,
    }

    response_tokens = []

    # Stream tokens output from NanoChat engine
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # Pop batch dimension (num_samples=1)
        response_tokens.append(token)

        # Decode token to text piece
        token_text = tokenizer.decode([token])

        # Yield raw text chunk immediately to client browser
        yield token_text

        # Yield control to the async event loop to send packet instantly
        await asyncio.sleep(0)

    # Ensure assistant_end token closes sequence in conversation history
    if response_tokens and response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)

    conversation_tokens.extend(response_tokens)

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_llm_stream(request.prompt),
        media_type="text/plain"
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "host": "sg.otiumcode.com",
        "port": 9010,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "ssl": True
    }

if __name__ == "__main__":
    print("Starting Otium NanoChat HTTPS Server on port 9010...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9010,
        ssl_certfile="fullchain.pem",
        ssl_keyfile="privkey.pem"
    )