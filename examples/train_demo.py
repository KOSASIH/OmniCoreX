"""
OmniCoreX Training Demo Script

Demonstrates a quickstart example for training the OmniCoreX model
using the prepared data loader and trainer modules.

Usage:
    python examples/train_demo.py
"""

import os
from utils import set_seed, load_config_file
from model import OmniCoreXModel
from data_loader import create_omncorex_dataloader
from trainer import Trainer

def dummy_tokenizer(text):
    # Simple tokenizer: ord mod 100 + 1
    return [ord(c) % 100 + 1 for c in text]

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    config = load_config_file(config_path)

    set_seed(config.get("seed", 42))

    # Prepare dataloader
    data_cfg = config["data"]
    train_loader = create_omncorex_dataloader(
        metadata_path=data_cfg["metadata_path"],
        modalities=data_cfg["modalities"],
        tokenizer=dummy_tokenizer,
        text_max_length=data_cfg.get("text_max_length", 256),
        image_size=tuple(data_cfg.get("image_size", [224, 224])),
        sensor_max_length=data_cfg.get("sensor_max_length", 100),
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=data_cfg.get("shuffle", True),
        num_workers=data_cfg.get("num_workers", 4),
        augmentation=data_cfg.get("augmentation", False)
    )

    # Initialize model
    model_cfg = config["model"]
    model = OmniCoreXModel(
        stream_configs=model_cfg["streams"],
        embed_dim=model_cfg.get("architecture", {}).get("embed_dim", 768),
        num_layers=model_cfg.get("architecture", {}).get("num_layers", 24),
        num_heads=model_cfg.get("architecture", {}).get("num_heads", 12),
        dropout=model_cfg.get("architecture", {}).get("dropout", 0.1)
    )

    # Trainer setup
    training_cfg = config.get("training", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=None,  # For demo, no validation loader
        save_dir=training_cfg.get("save_dir", "./checkpoints"),
        lr=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        accumulation_steps=training_cfg.get("accumulation_steps",1),
        total_steps=training_cfg.get("total_steps", 100000),
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        device=None,
        mixed_precision=training_cfg.get("mixed_precision", True)
    )

    # Start training for 1 epoch demo
    trainer.fit(epochs=1)

if __name__ == "__main__":
    main()

