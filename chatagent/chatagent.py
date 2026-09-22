import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Otium Code Chat Agent")

# Enable CORS so browser requests from otiumcode.com are accepted
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://otiumcode.com",
        "https://www.otiumcode.com",
        "http://localhost:8000", # Useful for local testing
        "*"                      # Change to specific origins in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

async def echo_stream(user_prompt: str):
    """
    Simulates streaming responses word-by-word.
    Replace this with your actual LLM GPU inference model!
    """
    response_text = f"Received prompt: '{user_prompt}'. Echoing from sg.otiumcode.com on port 9010."
    words = response_text.split(" ")

    for word in words:
        yield f"{word} "
        await asyncio.sleep(0.08)  # Simulates token generation delay

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        echo_stream(request.prompt),
        media_type="text/plain"
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "host": "sg.otiumcode.com", "port": 9010}

if __name__ == "__main__":
    # Listen on all network interfaces (0.0.0.0) on port 9010
    print("Starting Otium Chat Agent on port 9010...")
    
    # OPTION A: HTTP (if using Nginx/Caddy in front as an SSL reverse proxy)
    #uvicorn.run(app, host="0.0.0.0", port=9010)
    uvicorn.run(app, host="0.0.0.0", port=9010, ssl_certfile="fullchain.pem", ssl_keyfile="privkey.pem")

	# OPTION B: Direct HTTPS (if you generated Let's Encrypt certificates directly on the machine)
    # uvicorn.run(
    #     app, 
    #     host="0.0.0.0", 
    #     port=9010,
    #     ssl_keyfile="/etc/letsencrypt/live/sg.otiumcode.com/privkey.pem",
    #     ssl_certfile="/etc/letsencrypt/live/sg.otiumcode.com/fullchain.pem"
    # )
