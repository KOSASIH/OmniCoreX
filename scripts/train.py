#!/usr/bin/env python3
"""
OmniCoreX Training CLI Script

Launches the training process for OmniCoreX with command-line options
to configure parameters. Supports distributed and mixed precision training.
"""

import argparse
import os
from utils import set_seed, load_config_file, setup_logging
from model import OmniCoreXModel
from data_loader import create_omncorex_dataloader
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train OmniCoreX Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="Batch size override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config_file(args.config)
    if args.seed:
        set_seed(args.seed)
    else:
        set_seed(config.get("seed", 42))

    logger = setup_logging(log_file=args.log_file)
    logger.info("Starting OmniCoreX training")

    data_cfg = config["data"]
    training_cfg = config.get("training", {})

    batch_size = args.batch_size or data_cfg.get("batch_size", 16)
    mixed_precision = not args.no_mixed_precision and training_cfg.get("mixed_precision", True)

    train_loader = create_omncorex_dataloader(
        metadata_path=data_cfg["metadata_path"],
        modalities=data_cfg["modalities"],
        tokenizer=None,  # Assume custom tokenizer later
        batch_size=batch_size,
        shuffle=data_cfg.get("shuffle", True),
        num_workers=data_cfg.get("num_workers", 4),
        augmentation=data_cfg.get("augmentation", False)
    )

    model_cfg = config["model"]
    model = OmniCoreXModel(
        stream_configs=model_cfg["streams"],
        embed_dim=model_cfg.get("architecture", {}).get("embed_dim", 768),
        num_layers=model_cfg.get("architecture", {}).get("num_layers", 24),
        num_heads=model_cfg.get("architecture", {}).get("num_heads", 12),
        dropout=model_cfg.get("architecture", {}).get("dropout", 0.1)
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=None,
        save_dir=training_cfg.get("save_dir", "./checkpoints"),
        lr=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        accumulation_steps=training_cfg.get("accumulation_steps", 1),
        total_steps=training_cfg.get("total_steps", 100000),
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        device=None,
        mixed_precision=mixed_precision
    )

    epochs = args.epochs or training_cfg.get("epochs", 1)
    trainer.fit(epochs=epochs)

if __name__ == "__main__":
    main()
