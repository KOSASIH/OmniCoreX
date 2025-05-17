#!/usr/bin/env python3
"""
OmniCoreX Deployment Server CLI Script

Deploys OmniCoreX as a REST API server for real-time inference,
supporting asynchronous input processing and JSON responses.

Requirements:
- fastapi
- uvicorn

Usage:
    python scripts/deploy_server.py --config config.yaml --checkpoint ./checkpoints/checkpoint.pt --host 0.0.0.0 --port 8000
"""

import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from model import OmniCoreXModel
from utils import load_config_file, setup_logging

app = FastAPI()
model = None
tokenizer = None
device = None

class InferenceRequest(BaseModel):
    text: str
    # Extend with multimodal fields as needed, e.g., images encoded as base64

class InferenceResponse(BaseModel):
    response: str

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    global model, tokenizer, device
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_ids = tokenizer.encode(request.text) if tokenizer else [ord(c) for c in request.text]
        input_tensor = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            outputs = model(input_tensor)
        # Convert outputs to string, simplistic decoding:
        response_tokens = torch.argmax(outputs, dim=-1)[0].tolist()
        response_text = tokenizer.decode(response_tokens) if tokenizer else "".join(chr(t % 256) for t in response_tokens)
        return InferenceResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy OmniCoreX Model Serving API")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()

def main():
    global model, tokenizer, device
    args = parse_args()

    logger = setup_logging()
    config = load_config_file(args.config)

    model_cfg = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OmniCoreXModel(
        stream_configs=model_cfg["streams"],
        embed_dim=model_cfg.get("architecture", {}).get("embed_dim", 768),
        num_layers=model_cfg.get("architecture", {}).get("num_layers", 24),
        num_heads=model_cfg.get("architecture", {}).get("num_heads", 12),
        dropout=model_cfg.get("architecture", {}).get("dropout", 0.1)
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer if available - optionally from tokenizer.py
    try:
        from tokenizer import BPETokenizer
        # Assume vocabulary and merges paths from config or default location
        tokenizer = BPETokenizer()
        # tokenizer.load(<path_to_vocab>, <path_to_merges>)
    except ImportError:
        tokenizer = None
        logger.warning("Tokenizer module not found or failed to load.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
