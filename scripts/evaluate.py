#!/usr/bin/env python3
"""
OmniCoreX Evaluation CLI Script

Performs evaluation on a validation dataset, computing relevant metrics.
Supports loading checkpoints and configurable evaluation batch size.
"""

import argparse
import os
import torch
from utils import load_config_file, setup_logging
from model import OmniCoreXModel
from data_loader import create_omncorex_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OmniCoreX Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--log_file", type=str, help="Log file path")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config_file(args.config)
    logger = setup_logging(log_file=args.log_file)

    logger.info(f"Loading checkpoint from {args.checkpoint}")

    data_cfg = config["data"]
    eval_batch_size = args.batch_size or data_cfg.get("batch_size", 16)

    valid_loader = create_omncorex_dataloader(
        metadata_path=data_cfg["metadata_path"],
        modalities=data_cfg["modalities"],
        tokenizer=None,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        augmentation=False
    )

    model_cfg = config["model"]
    model = OmniCoreXModel(
        stream_configs=model_cfg["streams"],
        embed_dim=model_cfg.get("architecture", {}).get("embed_dim", 768),
        num_layers=model_cfg.get("architecture", {}).get("num_layers", 24),
        num_heads=model_cfg.get("architecture", {}).get("num_heads", 12),
        dropout=model_cfg.get("architecture", {}).get("dropout", 0.1)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    total_loss = 0.0
    count = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in valid_loader:
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**inputs)
            if "labels" in inputs:
                loss = criterion(outputs.view(-1, outputs.size(-1)), inputs["labels"].view(-1))
                total_loss += loss.item()
                count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    logger.info(f"Evaluation completed. Average loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
