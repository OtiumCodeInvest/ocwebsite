import sys
import os
import ast
import time
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("--- Hardware Diagnostics ---")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
print("----------------------------\n")

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

app = FastAPI(title="Otium Code NanoChat HTTPS Agent")

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
    expose_headers=["X-Total-Tokens", "X-Input-Tokens", "X-Output-Tokens", "X-Max-Tokens"]
)

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"

class ResetRequest(BaseModel):
    session_id: str

# -----------------------------------------------------------------------------
# Global LLM Model Initialization
# -----------------------------------------------------------------------------
print("Initializing NanoChat Model and Engine...")
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

model, tokenizer, meta = load_model("sft", device, phase="eval")

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

def safe_encode_special(tag: str):
    try:
        return tokenizer.encode_special(tag)
    except Exception:
        return None

python_start = safe_encode_special("<|python_start|>")
python_end = safe_encode_special("<|python_end|>")
calc_start = safe_encode_special("<|calculator_start|>")
calc_end = safe_encode_special("<|calculator_end|>")
output_start = safe_encode_special("<|output_start|>")
output_end = safe_encode_special("<|output_end|>")

engine = Engine(model, tokenizer)

# Sequence limits
MAX_CONTEXT_TOKENS = getattr(model.config, "sequence_len", 2048)
RESERVED_FOR_GENERATION = 768  # Set output buffer cap
MAX_PROMPT_BUDGET = MAX_CONTEXT_TOKENS - RESERVED_FOR_GENERATION  # 1280 input tokens

# Session store: session_id -> dict
sessions: Dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "tokens": [bos],
            "input_tokens_last": 0,
            "output_tokens_last": 0,
            "last_active": time.time()
        }
    sessions[session_id]["last_active"] = time.time()
    return sessions[session_id]

def reset_session(session_id: str):
    sessions[session_id] = {
        "tokens": [bos],
        "input_tokens_last": 0,
        "output_tokens_last": 0,
        "last_active": time.time()
    }

def truncate_to_context_budget(tokens: List[int], budget: int) -> List[int]:
    if len(tokens) <= budget:
        return tokens

    preserved = [tokens[0]]
    search_sub = tokens[1:]

    while len(search_sub) > budget and user_start in search_sub:
        try:
            next_turn = search_sub.index(user_start)
            following_turn = search_sub.index(user_start, next_turn + 1)
            search_sub = search_sub[following_turn:]
        except ValueError:
            break

    candidate = preserved + search_sub
    if len(candidate) > budget:
        candidate = [bos] + candidate[-(budget - 1):]

    return candidate

def get_model_parameters_count(m):
    return sum(p.numel() for p in m.parameters())

print(f"✅ NanoChat loaded on {device} (Max: {MAX_CONTEXT_TOKENS}, Input Budget: {MAX_PROMPT_BUDGET}, Gen Cap: {RESERVED_FOR_GENERATION})\n")

def safe_eval_math_expression(expr: str) -> str:
    expr = expr.strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()

    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd
    )
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "ERR"
        result = eval(compile(tree, filename='', mode='eval'), {"__builtins__": None}, {})
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception:
        return "ERR"

# -----------------------------------------------------------------------------
# Streaming Token Generator
# -----------------------------------------------------------------------------
async def generate_llm_stream(user_input: str, session_id: str, request: Request):
    clean_input = user_input.strip()
    if not clean_input:
        return

    sess = get_session(session_id)
    tokens = sess["tokens"]
    prior_turn_count = len(tokens)

    # 1. Prompt Truncation & Budget Protection
    new_prompt_tokens = tokenizer.encode(clean_input)
    prompt_len = len(new_prompt_tokens)
    max_single_budget = MAX_PROMPT_BUDGET - 4

    if prompt_len > max_single_budget:
        print(f"\n[⚠️ TRUNCATED INPUT] Session '{session_id}': input had {prompt_len} tokens, sliced to {max_single_budget}.")
        new_prompt_tokens = new_prompt_tokens[-max_single_budget:]

    tokens.append(user_start)
    tokens.extend(new_prompt_tokens)
    tokens.append(user_end)
    tokens.append(assistant_start)

    tokens = truncate_to_context_budget(tokens, MAX_PROMPT_BUDGET)
    sess["tokens"] = tokens
    sess["input_tokens_last"] = len(tokens)

    max_gen_tokens = min(RESERVED_FOR_GENERATION, MAX_CONTEXT_TOKENS - len(tokens))

    print(f"\n[Prompt Input] Session '{session_id}' | Prior Context: {prior_turn_count} toks | "
          f"Prompt: {len(new_prompt_tokens)} toks | Context After Input: {len(tokens)}/{MAX_CONTEXT_TOKENS} | "
          f"Generation Cap: {max_gen_tokens} toks")

    if max_gen_tokens <= 0:
        yield "Error: Context limit reached."
        return

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": max_gen_tokens,
        "temperature": 0.6,
        "top_k": 50,
    }

    response_tokens = []
    current_context = list(tokens)
    in_calc_mode = False
    calc_token_buffer = []
    max_tool_executions = 6
    tool_counter = 0
    natural_end = False

    try:
        while True:
            generator = engine.generate(current_context, **generate_kwargs)

            for token_column, token_masks in generator:
                if await request.is_disconnected():
                    print(f"\n[Client Disconnect] Session '{session_id}' aborted mid-stream.")
                    return

                token = token_column[0]
                response_tokens.append(token)
                current_context.append(token)

                token_text = tokenizer.decode([token])
                yield token_text
                await asyncio.sleep(0)

                if token == assistant_end:
                    natural_end = True
                    break

                if token in (python_start, calc_start):
                    in_calc_mode = True
                    calc_token_buffer = []
                    continue

                if in_calc_mode:
                    if token in (python_end, calc_end):
                        in_calc_mode = False
                        tool_counter += 1

                        math_expr = tokenizer.decode(calc_token_buffer)
                        eval_result = safe_eval_math_expression(math_expr)

                        if output_start is not None and output_end is not None:
                            tool_output_str = f"<|output_start|>{eval_result}<|output_end|>"
                            output_tokens = (
                                [output_start]
                                + tokenizer.encode(eval_result)
                                + [output_end]
                            )
                        else:
                            tool_output_str = f"\nOutput: {eval_result}\n"
                            output_tokens = tokenizer.encode(tool_output_str)

                        yield tool_output_str
                        await asyncio.sleep(0)

                        response_tokens.extend(output_tokens)
                        current_context.extend(output_tokens)

                        if tool_counter < max_tool_executions:
                            break
                    else:
                        calc_token_buffer.append(token)

            if response_tokens and (response_tokens[-1] == assistant_end or tool_counter >= max_tool_executions):
                break

            if not in_calc_mode:
                break

    except (asyncio.CancelledError, GeneratorExit):
        print(f"\n[Stream Cancelled] Session '{session_id}'")
    except Exception as e:
        print(f"\n[Stream Error] Session '{session_id}': {e}")
    finally:
        gen_count = len(response_tokens)
        sess["output_tokens_last"] = gen_count
        if natural_end:
            status_str = f"✅ Finished naturally in {gen_count} toks."
        elif gen_count >= max_gen_tokens:
            status_str = f"⚠️ CUTOFF: Hit generation cap ({gen_count}/{max_gen_tokens} toks)."
        else:
            status_str = f"Stopped after {gen_count} toks."

        print(f"[Generation Status] Session '{session_id}' | {status_str} | Total Session Context: {len(tokens) + gen_count}/{MAX_CONTEXT_TOKENS}")

        if response_tokens:
            if response_tokens[-1] != assistant_end:
                response_tokens.append(assistant_end)
            tokens.extend(response_tokens)
            sess["tokens"] = tokens

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    session_id = chat_req.session_id or "default"

    # Catch clear command from collector
    if chat_req.prompt.strip().lower() == "clear":
        reset_session(session_id)
        sess = get_session(session_id)
        return JSONResponse({"status": "cleared", "session_id": session_id, "used_tokens": len(sess["tokens"])})

    sess = get_session(session_id)
    current_tokens = len(sess["tokens"])

    return StreamingResponse(
        generate_llm_stream(chat_req.prompt, session_id, request),
        media_type="text/plain",
        headers={
            "X-Total-Tokens": str(current_tokens),
            "X-Input-Tokens": str(sess["input_tokens_last"]),
            "X-Output-Tokens": str(sess["output_tokens_last"]),
            "X-Max-Tokens": str(MAX_CONTEXT_TOKENS)
        }
    )

@app.get("/api/info")
async def info_endpoint(session_id: Optional[str] = "default"):
    sess = get_session(session_id)
    cfg = getattr(model, "config", None)

    return {
        "model": {
            "name": "NanoChat SFT",
            "parameters": get_model_parameters_count(model),
            "layers": getattr(cfg, "n_layer", getattr(cfg, "depth", "Unknown")),
            "dim": getattr(cfg, "n_embd", getattr(cfg, "model_dim", "Unknown")),
            "heads": getattr(cfg, "n_head", getattr(cfg, "num_heads", "Unknown")),
            "vocab_size": getattr(cfg, "vocab_size", getattr(tokenizer, "n_words", getattr(tokenizer, "vocab_size", 32768))),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "limits": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "input_budget": MAX_PROMPT_BUDGET,
            "reserved_for_generation": RESERVED_FOR_GENERATION
        },
        "session": {
            "session_id": session_id,
            "total_used": len(sess["tokens"]),
            "input_used": sess["input_tokens_last"],
            "output_used": sess["output_tokens_last"]
        }
    }

@app.post("/api/reset")
async def reset_endpoint(body: ResetRequest):
    reset_session(body.session_id)
    return {"status": "cleared", "session_id": body.session_id, "used_tokens": 1}

@app.get("/api/token_usage")
async def token_usage(session_id: str):
    sess = get_session(session_id)
    return {
        "used_tokens": len(sess["tokens"]),
        "input_used": sess["input_tokens_last"],
        "output_used": sess["output_tokens_last"],
        "max_tokens": MAX_CONTEXT_TOKENS,
        "input_limit": MAX_PROMPT_BUDGET,
        "output_limit": RESERVED_FOR_GENERATION
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9010,
        ssl_certfile="fullchain.pem",
        ssl_keyfile="privkey.pem",
        timeout_keep_alive=30
    )